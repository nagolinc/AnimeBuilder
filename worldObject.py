import re


class WorldObject:
    def __init__(
            self,
            templates,
            textGenerator,
            objectName,
            objects=None,
            genTextAmount_min=15,
            genTextAmount_max=30,
            no_repeat_ngram_size=8,
            repetition_penalty=2.0,
            MIN_ABC=4,
            num_beams=8,
            temperature=1.0,
            verbose=False):

        self.textGenerator = textGenerator

        self.verbose = verbose

        self.temperature = temperature
        self.MIN_ABC = MIN_ABC
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams

        self.genTextAmount_min = genTextAmount_min
        self.genTextAmount_max = genTextAmount_max
        self.MAX_DEPTH = 5

        self.objectName = objectName
        self.templates = templates
        if objects is None:
            self.objects = {}
        else:
            self.objects = objects
        self.filledTemplate = self.fillTemplate(templates[objectName])

        self.hiddenStates = None

        if self.verbose:
            print("GOT FILLED TEMPLATE", self.objectName,
                  "\n\n", self.filledTemplate, "\n\n")

        self.object = self.parseTemplate(self.filledTemplate)

    def generateTextWithInput(self, textInput, depth=0):

        if depth > self.MAX_DEPTH:
            return "error"

        input_ids = self.textGenerator['tokenizer'](
            textInput, return_tensors="pt").input_ids
        amt = input_ids.shape[1]
        result = self.textGenerator['pipeline'](
            textInput,
            do_sample=True,
            # gah, this is in units of tokens, not chars
            min_length=amt+self.genTextAmount_min,
            max_length=amt+self.genTextAmount_max,
            # max_new_tokens=self.genTextAmount,
            # max_length=None,
            pad_token_id=50256,
            return_full_text=False,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
            num_beams=self.num_beams,
            temperature=self.temperature
        )

        lines = result[0]['generated_text'].strip().split("\n")

        '''

        print('about to die',textInput)

        input_ids = self.textGenerator['tokenizer'](
            textInput, return_tensors="pt").input_ids.to("cuda")

        amt = len(input_ids)

        outputs = self.textGenerator['model'].generate(input_ids, do_sample=True,
                                                       min_length=amt+self.genTextAmount,  # gah, this is in units of tokens, not chars
                                                       max_length=amt+self.genTextAmount,
                                                       pad_token_id=50256,
                                                       no_repeat_ngram_size=self.no_repeat_ngram_size,
                                                       repetition_penalty=self.repetition_penalty,
                                                       num_beams=self.num_beams,
                                                       temperature=self.temperature
                                                       )

        result = self.textGenerator['tokenizer'].batch_decode(
            outputs, skip_special_tokens=True)

        lines=result[0]['generated_text'].strip().split("\n")
        '''

        # remove len()==0 lines
        lines = [line.strip() for line in lines if len(line.strip()) > 0]
        # make sure we have at least some output
        if len(lines) == 0:
            if self.verbose:
                print('no response', result, textInput)
            return self.generateTextWithInput(textInput, depth=depth+1)
        rv = lines[0]
        # remove non-ascii
        rv = rv.encode("ascii", errors="ignore").decode()

        if rv[:3] == "ick":
            print(textInput, result, rv)
            assert False

        # remove trailing ":"s
        if rv[-1] == ":":
            if self.verbose:
                print('trailing :', result)
            return self.generateTextWithInput(textInput, depth=depth+1)
        # ":"s should actually just never appear
        if ":" in rv:
            if self.verbose:
                print(': present', result)
            return self.generateTextWithInput(textInput, depth=depth+1)
        # anything that's all punctuation is also bad
        #rva = re.sub(r'\W+', '', rv)
        rva = re.sub(r'[^a-zA-Z]+', '', rv)
        if len(rva) < self.MIN_ABC:
            if self.verbose:
                print('non alphanumeric', result, self.MIN_ABC, depth=depth+1)
            return self.generateTextWithInput(textInput)

        return rv

    def fillTemplate(self, template):
        t = 0
        output = ""
        thisMatch = re.search("{[^}]*}", template)
        while thisMatch:
            start, end = thisMatch.span()
            obj_and_prop = template[t+start+1:t+end-1]

            output += template[t:t+start]

            gotProp = self.getObjwithProp(obj_and_prop, output)

            output += str(gotProp)

            if self.verbose:
                print("MATCH", thisMatch, gotProp)

            t = t+end
            thisMatch = re.search("{[^}]*}", template[t:])

        output += template[t:]

        return output

    def parseTemplate(self, template):

        # clean up whitespace
        template = "\n".join([line.strip() for line in template.split("\n")])

        objects = template.split("\n\n")
        if self.verbose:
            print(objects)
        objects = [o for o in objects if len(o) > 0]
        thisObject = objects[-1]
        output = {}
        propName = "NONE"
        for i, line in enumerate(thisObject.split("\n")):
            line = line.strip()
            # print(i,line)
            if line.endswith(":"):
                # print("here0")
                propName = line[:-1]
            else:
                #print("here1, propName=",propName)
                if propName != "NONE" and len(line) > 0:
                    if propName in output:
                        output[propName] += "\n"+line
                    else:
                        output[propName] = line
        return output

    def getObjwithProp(self, obj_and_prop, output):

        overrides = None
        objType = None
        # handle ":"s
        if ":" in obj_and_prop:
            obj_and_prop, objType, overrides = obj_and_prop.split(":")

        # handle "."s
        propName = None
        if "." in obj_and_prop:
            objectName, propName = obj_and_prop.split(".")
        else:
            objectName = obj_and_prop

        # handle saved objects
        if objectName in self.objects:
            thisObject = self.objects[objectName]
            if propName is not None:
                return thisObject.getProperty(propName)
            else:
                return thisObject

        # handle type text
        if objType == "TEXT" or obj_and_prop == "TEXT":
            if self.verbose == 2:
                print("generating text", objType,
                      obj_and_prop, "with template", output)

            output = output.strip()  # remove trailing " "s
            #output = self.generateTextWithInput(output)
            text = self.generateTextWithInput(output)
            if objectName != "TEXT":
                self.objects[objectName] = text
            return text
        else:
            if self.verbose:
                print("got prop", objectName, propName)
            thisObject = self.getObject(objectName, objType, overrides)
            if propName is not None:
                return thisObject.getProperty(propName)
            else:
                return thisObject

    def getObject(self, objectName, objType, overrides=None):
        if objectName in self.objects:
            return self.objects[objectName]
        else:
            # handle overrides
            objects = None
            if overrides:
                # parse overrides "a=b,c=d,..."
                objects = {}
                for override in overrides.split(","):
                    for k, v in override.split("="):
                        objects[k] = self.getObject(v)
            # remove trailing digits
            if objType is None:
                objType = re.sub(r'\d+$', '', objectName)
            # generate object
            thisObject = WorldObject(self.templates, self.textGenerator, objType, objects=objects,
                                     genTextAmount_min=self.genTextAmount_min,
                                     genTextAmount_max=self.genTextAmount_max,
                                     no_repeat_ngram_size=self.no_repeat_ngram_size,
                                     repetition_penalty=self.repetition_penalty,
                                     MIN_ABC=self.MIN_ABC,
                                     num_beams=self.num_beams,
                                     temperature=self.temperature,
                                     verbose=self.verbose)
            # store for future use
            self.objects[objectName] = thisObject
            return self.objects[objectName]

    def getProperty(self, propName):
        if self.verbose:
            print("getting property", propName, "from object", self.object)
        if propName in self.objects:
            return self.objects[propName]
        if propName in self.object:
            return self.object[propName]
        raise ValueError("property not found!")

    def __repr__(self):
        s = self.filledTemplate.split("\n\n")
        # remove empty lines
        v = ["\n".join([line for line in lines.split(
            "\n") if len(line.strip()) > 0]) for lines in s]
        v = [x for x in v if len(x) > 0]
        r = v[-1]
        return "<world object:%s>\n" % self.objectName+r

    def __str__(self):
        try:
            return str(self.getProperty("description")).strip()
        except:
            return self.__repr__()
