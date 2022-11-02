import re

class WorldObject:
    def __init__(self,templates,textGenerator,objectName,objects=None,genTextAmount=20,verbose=False):

        self.textGenerator=textGenerator
        
        self.verbose=verbose
        
        self.MIN_ABC=4
        
        self.genTextAmount=genTextAmount
        
        self.objectName=objectName
        self.templates=templates
        if objects is None:
            self.objects={}
        else:
            self.objects=objects
        self.filledTemplate=self.fillTemplate(templates[objectName])
        
        self.hiddenStates=None
        
        
        
        
        if self.verbose:
            print("GOT FILLED TEMPLATE",self.objectName,"\n\n",self.filledTemplate,"\n\n")
        
        self.object=self.parseTemplate(self.filledTemplate)        
        
        
    def generateTextWithInput(self,textInput,doContinue=False):
        
        if doContinue:
            hiddenStates=self.hiddenStates
        else:
            hiddenStates=None
        
        
        
        result=self.textGenerator(textInput, do_sample=True, 
                          max_new_tokens=self.genTextAmount,
                         pad_token_id=50256,
                         return_full_text=False,
                         no_repeat_ngram_size=8,
                         repetition_penalty=2.0,
                         max_length=None,
                        )
        
        lines=result[0]['generated_text'].strip().split("\n")
        
        #remove len()==0 lines
        lines=[line.strip() for line in lines if len(line.strip())>0]
        #make sure we have at least some output
        if len(lines)==0:
            return self.generateTextWithInput(textInput)
        rv=lines[0]
        #remove trailing ":"s
        if rv[-1]==":":
            return self.generateTextWithInput(textInput)
        #":"s should actually just never appear
        if ":" in rv:
            return self.generateTextWithInput(textInput)
        #anything that's all punctuation is also bad
        rva = re.sub(r'\W+', '', rv)
        if len(rva)<self.MIN_ABC:
            return self.generateTextWithInput(textInput)
        
        return rv
        
    def fillTemplate(self,template):
        t=0
        output=""
        thisMatch=re.search("{[^}]*}", template)
        while thisMatch:
            start,end=thisMatch.span()
            obj_and_prop=template[t+start+1:t+end-1]            
            
            output+=template[t:t+start]
            
            gotProp=self.getObjwithProp(obj_and_prop,output)
            
            if isinstance(gotProp, str):
                output+=gotProp
            else:
                output+=gotProp.getProperty("description")
                
            if self.verbose:
                print("MATCH",thisMatch,gotProp)
                
            t=t+end
            thisMatch=re.search("{[^}]*}", template[t:])
        
        output+=template[t:]

        return output
        
        
    def parseTemplate(self,template):
        objects=template.split("\n\n")
        if self.verbose:
            print(objects)
        objects=[o for o in objects if len(o)>0]
        thisObject=objects[-1]
        output={}
        propName="NONE"
        for i,line in enumerate(thisObject.split("\n")):
            line=line.strip()
            #print(i,line)
            if line.endswith(":"):
                #print("here0")
                propName=line[:-1]
            else:
                #print("here1, propName=",propName)
                if propName != "NONE" and len(line)>0:
                    if propName in output:
                        output[propName]+="\n"+line
                    else:
                        output[propName]=line
        return output
    
    
    def getObjwithProp(self,obj_and_prop,output):
          
            overrides=None
            objType=None
            #handle ":"s
            if ":" in obj_and_prop:
                obj_and_prop,objType,overrides=obj_and_prop.split(":")
            
            #handle "."s
            propName=None
            if "." in obj_and_prop:
                objectName,propName=obj_and_prop.split(".")               
            else:
                objectName=obj_and_prop
                    
            #handle saved objects    
            if objectName in self.objects:
                thisObject = self.objects[objectName]                    
                if propName is not None:
                    return thisObject.getProperty(propName)
                else:
                    return thisObject
                
                
                
            #handle type text       
            if objType=="TEXT" or obj_and_prop=="TEXT":
                if self.verbose==2:
                    print("generating text",objType,obj_and_prop,"with template",output)
                text= self.generateTextWithInput(output)
                if objectName!="TEXT":
                    self.objects[objectName]=text
                return text
            else:          
                if self.verbose:
                    print("got prop",objectName,propName)
                thisObject=self.getObject(objectName,objType,overrides)
                if propName is not None:
                    return thisObject.getProperty(propName)
                else:
                    return thisObject

        
    def getObject(self,objectName,objType,overrides=None):
        if objectName in self.objects:
            return self.objects[objectName]
        else:
            #handle overrides
            objects=None        
            if overrides:
                #parse overrides "a=b,c=d,..."
                objects={}
                for override in overrides.split(","):
                    for k,v in override.split("="):
                        objects[k]=self.getObject(v)  
            #remove trailing digits
            if objType is None:
                objType= re.sub(r'\d+$', '', objectName)
            #generate object
            thisObject=WorldObject(self.templates,self.textGenerator,objType,objects=objects,verbose=self.verbose)
            #store for future use
            self.objects[objectName]=thisObject
            return self.objects[objectName]
        
    def getProperty(self,propName):
        if self.verbose:
            print("getting property",propName,"from object",self.object)
        if propName in self.objects:
            return self.objects[propName]
        if propName in self.object:
            return self.object[propName]
        raise ValueError("property not found!")
        
        
    def __repr__(self):
        s=self.filledTemplate.split("\n\n")
        #remove empty lines
        v=["\n".join([line for line in lines.split("\n") if len(line.strip())>0]) for lines in s]
        v=[x for x in v if len(x)>0]
        r=v[-1]        
        return "<world object:%s>\n"%self.objectName+r
    
    
        
            
            
            
            
            
        
        
        
        