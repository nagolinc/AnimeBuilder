import os
import openai
            

import re
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM

import gc
import torch
import random
from PIL import Image
import urllib
from pydub import AudioSegment
from io import BytesIO
import numpy as np
#from IPython.display import Audio, display
from ipywidgets import Audio  # no good, doesn't stop when clear display

import ipywidgets as widgets
from torch import autocast
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from diffusers.models import AutoencoderKL


from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

from mubert import generate_track_by_prompt
from templates import templates
from worldObject import WorldObject, ListObject

import riffusion
from riffusion import get_music

import logging
import uuid


class AnimeBuilder:

    def __init__(
        self,
        textModel='EleutherAI/gpt-neo-1.3B',
        diffusionModel="hakurei/waifu-diffusion",
        vaeModel="stabilityai/sd-vae-ft-mse",
        templates=templates,
        advanceSceneObjects=None,
        num_inference_steps=30,
        cfg=None,
        verbose=False,
        doImg2Img=False,
        saveMemory=False,
        cache_dir='e:/img/hf',
        textRevision=None,
        negativePrompt="collage, grayscale, text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body",
        suffix=", anime drawing",
        riffusionSuffix=" pleasing rythmic background music",
        savePath="./static/samples/",
        saveImages=False
    ):
        self.savePath=savePath
        self.saveImages=saveImages
        
        self.textModel=textModel

        self.cache_dir=cache_dir

        self.saveMemory = saveMemory

        self.verbose = verbose

        self.mubert = False

        self.templates = templates

        if cfg is None:
            cfg = {
                "genTextAmount_min": 30,
                "genTextAmount_max": 100,
                "no_repeat_ngram_size": 8,
                "repetition_penalty": 2.0,
                "MIN_ABC": 4,
                "num_beams": 8,
                "temperature": 1.0,
                "MAX_DEPTH": 5
            }
        self.cfg = cfg

        self.num_inference_steps = num_inference_steps

        self.negativePrompt=negativePrompt
        self.suffix=suffix
        self.riffusionSuffix=riffusionSuffix

        # use this for advanceScene()
        # advance scene
        if advanceSceneObjects is None:
            self.advanceSceneObjects = [
                {
                    "object": "advancePlot",
                    "whichScene": 3,
                    "numScenes": 3,
                },
                {
                    "object": "fightScene",
                    "whichScene": 1,
                    "numScenes": 3,
                },
            ]
        else:
            self.advanceSceneObjects = advanceSceneObjects

        if self.verbose:
            print("LOADING TEXT MODEL")

        
        if textModel=="GPT3":
            pass
            #self.textGenerator="GPT3"

            self.textGenerator = {
                'name':"GPT3",
            }

            openai.organization = "org-bKm1yrKncCnPfkcf8pDpe4GM"
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.Model.list()
        elif textModel=="GPT3-turbo":
            #self.textGenerator="GPT3-turbo"
            self.textGenerator = {
                'name':"GPT3-turbo",
            }

            openai.organization = "org-bKm1yrKncCnPfkcf8pDpe4GM"
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.Model.list()
        else:        
            # text model
            self.textModel = textModel
            self.textRevision = textRevision

            textGenerator = pipeline('text-generation',
                                    revision=self.textRevision,
                                    torch_dtype=torch.float16,
                                    model=self.textModel,
                                    device=0
                                    #device_map="auto",
                                    )
            

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.textModel, torch_dtype=torch.float16)
            # self.textModel = AutoModelForCausalLM.from_pretrained(
            #    self.textModel, torch_dtype=torch.float16).to('cuda')

            '''

            from accelerate import init_empty_weights
            from transformers import AutoConfig, AutoModelForCausalLM

            checkpoint = "EleutherAI/gpt-j-6B"
            config = AutoConfig.from_pretrained(checkpoint)

            with init_empty_weights():
                self.textModel = AutoModelForCausalLM.from_config(config)

            from accelerate import load_checkpoint_and_dispatch

            from accelerate import infer_auto_device_map

            device_map = infer_auto_device_map(self.textModel, max_memory={0: "12GiB", "cpu": "12GiB"})

            self.textModel = load_checkpoint_and_dispatch(
                self.textModel, "sharded-gpt-j-6B", device_map=device_map, no_split_module_classes=["GPTJBlock"]
            )

            '''

            self.textGenerator = {
                'name':self.textModel,
                'tokenizer': self.tokenizer,
                #'model': self.textModel
                'pipeline': textGenerator
            }

        # image model

        if self.verbose:
            print("LOADING IMAGE MODEL")

        # scheduler
        scheduler = DPMSolverMultistepScheduler.from_config(
            diffusionModel, subfolder='scheduler')

        

        # make sure you're logged in with `huggingface-cli login`
        vae = AutoencoderKL.from_pretrained(vaeModel)

        '''
        self.pipe = StableDiffusionPipeline.from_pretrained(
            diffusionModel,
            scheduler=scheduler,
            vae=vae,
            # revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True,
            cache_dir="./AI/StableDiffusion")

        self.pipe.safety_checker = None

        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()

        

        scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler=scheduler

        '''

        pipe = StableDiffusionPipeline.from_pretrained(diffusionModel,vae=vae, torch_dtype=torch.float16)

        # change to UniPC scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()

        self.pipe=pipe

        self.pipe.safety_checker = None

        self.doImg2Img = doImg2Img

        if self.doImg2Img:
            if self.verbose:
                print("LOADING Img2Img")
            self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                diffusionModel,
                # revision=revision,
                scheduler=self.pipe.scheduler,
                unet=self.pipe.unet,
                vae=self.pipe.vae,
                safety_checker=self.pipe.safety_checker,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                torch_dtype=torch.float16,
                use_auth_token=True,
                cache_dir="./AI/StableDiffusion"
            )

            self.img2img.enable_attention_slicing()
            self.img2img.enable_xformers_memory_efficient_attention()

        if self.verbose:
            print("LOADING TTS MODEL")

        # tts
        #

        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-200_speaker-cv4",  # random
            # "Voicemod/fastspeech2-mf4",#morgan freedman
            # "Voicemod/fastspeech2-en-male1",#english male
            # "Voicemod/fastspeech2-en-ljspeech",#female
            # 'facebook/tts_transformer-es-css10',#spanish male, doesn't work for english
            arg_overrides={"vocoder": "hifigan", "fp16": False, }
        )

        self.tts_models = models
        self.tts_cfg = cfg
        self.tts_task = task

        #model = models[0]
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        self.tts_generator = task.build_generator(models, cfg)

        #  000000000011111111112222222222333333333344444444444555555555
        #  012345678901234567890123456789012345678901234567890123456789
        s = "FMFMMMMMFMMMFFMFFMMMMMMmffmmfmmfmfmmmmmmfmmmmmfmmmffmmmm".upper()
        self.maleVoices = [i for i in range(len(s)) if s[i] == "M"]
        self.femaleVoices = [i for i in range(len(s)) if s[i] == "F"]

    def doGen(self, prompt, num_inference_steps=30, recursion=0):

        # move text model to cpu for now
        if self.saveMemory:
            self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cpu(
            )
            gc.collect()
            torch.cuda.empty_cache()

        with autocast("cuda"):
            image = self.pipe([prompt],
                              negative_prompt=[self.negativePrompt],
                              guidance_scale=7.5,
                              num_inference_steps=num_inference_steps,
                              width=512,
                              height=512).images[0]

        if self.doImg2Img:
            img2Input = image.resize((1024, 1024))
            with autocast("cuda"):
                img2 = self.img2img(
                    prompt=prompt,
                    image=img2Input,
                    strength=0.25,
                    guidance_scale=7.5,
                    num_inference_steps=num_inference_steps,
                ).images[0]
                output = img2
        else:
            output = image

        if self.saveMemory:
            gc.collect()
            torch.cuda.empty_cache()

            #self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cuda()

        # fix all black images? (which Anything 3.0 puts out sometimes)
        pix = np.array(output)
        MAX_IMG_RECURSION = 3
        if np.sum(pix) == 0 and recursion < MAX_IMG_RECURSION:
            if self.verbose:
                print("REDOING BLANK IMAGE!")
            return self.doGen(prompt, num_inference_steps, recursion=recursion+1)

        return output

    def textToSpeech(self, text, voice):

        mp3file_name=self.savePath+str(uuid.uuid4())+".mp3"
        wavfile_name=self.savePath+str(uuid.uuid4())+".wav"

        try:
            with autocast("cuda"):
                self.tts_task.data_cfg.hub["speaker"] = voice
                sample = TTSHubInterface.get_model_input(self.tts_task, text)
                #print("about to die",models[0],sample)
                wav, rate = TTSHubInterface.get_prediction(
                    self.tts_task, self.tts_models[0], self.tts_generator, sample)
                # print("huh?",wav,rate,len(wav)/rate)
                duration = len(wav)/rate

            audio=ipd.Audio(wav.cpu(), rate=rate, autoplay=True)
            with open(wavfile_name, 'wb') as f:
                f.write(audio.data)

            wavfile = AudioSegment.from_wav(wavfile_name)
            wavfile.export(mp3file_name, format="mp3")

            return mp3file_name, duration
        except:
            print("Error generating text", text, voice)
    # music

    def generate_track_by_prompt_vol(self, prompt, vol=1.0, duration=8, loop=True, autoplay=True):

        mp3file_name=self.savePath+str(uuid.uuid4())+".mp3"
        wavfile_name=self.savePath+str(uuid.uuid4())+".wav"

        if self.mubert:

            url = generate_track_by_prompt(prompt, duration, loop)
            if url is None:
                return
            mp3 = urllib.request.urlopen(url).read()
            original = AudioSegment.from_mp3(BytesIO(mp3))
            samples = original.get_array_of_samples()
            samples /= np.max(np.abs(samples))
            samples *= vol
            # audio = Audio(samples, normalize=False,
            #              rate=original.frame_rate, autoplay=autoplay)

            #audio = Audio.from_file("audio.mp3", loop=True, autoplay=True)

            #return audio
            return mp3file_name
        else:
            _, filename = get_music(prompt+self.riffusionSuffix, duration, wavfile_name=wavfile_name, mp3file_name=mp3file_name)
            mp3 = open(filename, 'rb').read()
            original = AudioSegment.from_mp3(BytesIO(mp3))
            samples = original.get_array_of_samples()
            samples /= np.max(np.abs(samples))
            samples *= vol
            # audio = Audio(samples, normalize=False,
            #              rate=original.frame_rate, autoplay=autoplay)
            #audio = Audio.from_file("audio.mp3", loop=True, autoplay=True)

            #return audio
            return mp3file_name

    def descriptionToCharacter(self, description):
        thisObject = WorldObject(self.templates, self.textGenerator, "descriptionToCharacter", objects={
            "description": description},
            cfg=self.cfg,
            verbose=self.verbose
        )
        return thisObject

    def advanceStory(self, story, subplot, mainCharacter=None, supportingCharacters=None, alwaysUseMainCharacter=True):

        #save some memory
        self.pipe.to("cpu")
        riffusion.pipe2.to('cpu')

        gc.collect()
        torch.cuda.empty_cache()


        advanceSceneObject = random.choice(self.advanceSceneObjects)

        # update subplot

        if alwaysUseMainCharacter:
            character1 = mainCharacter
            character2 = random.choice(supportingCharacters)
        else:
            character1, character2 = random.sample(
                [mainCharacter]+supportingCharacters, 2)

        if character1 is None:
            character1 = story.getProperty("character1")
        if character2 is None:
            character2 = story.getProperty("character2")

        newStory = WorldObject(self.templates, self.textGenerator, advanceSceneObject['object'], objects={
            "character1": character1,
            "character2": character2,
            "previous": story,
        },
            cfg=self.cfg,
            verbose=self.verbose
        )

        whichScene = advanceSceneObject['whichScene']
        numScenes = advanceSceneObject['numScenes']

        self.pipe.to("cuda")
        riffusion.pipe2.to('cuda')

        gc.collect()
        torch.cuda.empty_cache()

        return whichScene, numScenes, newStory

    def sceneToTranscript(self, scene, k=3, character1=None, character2=None, whichScene=1):
        if character1 is None:
            character1 = scene.getProperty("character1")
        if character2 is None:
            character2 = scene.getProperty("character2")

        objects = {"story synopsis": scene.getProperty("story synopsis"),
                   "subplot": scene.getProperty("subplot"),
                   "scene": scene.getProperty("scene %d" % k),
                   "character1": character1,
                   "character2": character2,
                   }

        # check for dialogue
        line1txt = None
        try:
            line1txt = scene.getProperty("scene %d line 1 text" % whichScene)
            if self.verbose:
                print("line 1 text", line1txt)
        except:
            if self.verbose:
                print("no property", "scene %d line 1 text" % whichScene)
            pass
        if line1txt:
            objects['line 1 text'] = line1txt

        thisObject = WorldObject(self.templates, self.textGenerator,
                                 "sceneToTranscript", objects,
                                 cfg=self.cfg,
                                 verbose=self.verbose
                                 )
        return thisObject

    def watchAnime(
        self,
        synopsis=None,
        subplot1=None,
        scene1=None,
        character1=None,
        num_characters=4,
        k=100,
        amtMin=15,
        amtMax=30,
        promptSuffix="",
        portrait_size=128,
        skip_transcript=False,
        whichScene=1,  # optionally skip first few scenes
        alwaysUseMainCharacter=True,  # always use main character in scene
    ):

        # make sure text generator is on cuda (can get out of sync if we ctrl+c during doGen() )
        if self.textGenerator["name"].startswith("GPT3"):
            self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cuda()

        self.amtMin = amtMin
        self.amtMax = amtMax

        objects = {}
        if synopsis:
            objects['story synopsis'] = synopsis
        if scene1:
            objects['scene 1 text'] = scene1
        if character1:
            if isinstance(character1, str):
                character1 = self.descriptionToCharacter(character1)
                # print(character1)
        else:
            character1 = WorldObject(self.templates, self.textGenerator, "character",
                                     cfg=self.cfg,
                                     verbose=self.verbose
                                     )

        mainCharacter = character1
        objects['character1'] = mainCharacter
        if self.verbose:
            print("main character", mainCharacter.__repr__())

        names = set()
        names.add(str(mainCharacter.getProperty("name")))
        # generate characters
        supportingCharacters = []
        while len(supportingCharacters) < num_characters-1:
            newCharacter = WorldObject(self.templates, self.textGenerator, "character",
                                       cfg=self.cfg,
                                       verbose=self.verbose
                                       )
            thisName = str(newCharacter.getProperty("name"))
            if thisName not in names:
                if self.verbose:
                    print(newCharacter.__repr__())
                supportingCharacters += [newCharacter]
                names.add(thisName)
            else:
                if self.verbose:
                    print("skipping repeated character", thisName)

        if subplot1:
            objects['part 1'] = subplot1

        for i in range(3):
            objects['character%d' % (i+2)] = supportingCharacters[i]

        plotOverview = WorldObject(
            self.templates, self.textGenerator, "plot overview",
            cfg=self.cfg,
            verbose=self.verbose
        )

        subplot = plotOverview.getProperty("part 1")
        objects["subplot"] = subplot

        story = WorldObject(self.templates, self.textGenerator,
                            "storyWithCharacters",
                            cfg=self.cfg,
                            objects=objects,
                            verbose=self.verbose
                            )
        if self.verbose:
            print(story)

        # get voices
        voices = {}
        genders = {}
        for thisCharacter in [mainCharacter]+supportingCharacters:
            name = str(thisCharacter.getProperty("name"))
            gender = thisCharacter.getProperty("gender")
            if gender == "male":
                voices[name] = random.choice(self.maleVoices)
            else:
                voices[name] = random.choice(self.femaleVoices)
            genders[name] = gender
            description = thisCharacter.getProperty("description")

        # generate portraits
        portraits = {}
        for thisCharacter in [mainCharacter]+supportingCharacters:
            name = str(thisCharacter.getProperty("name"))
            gender = thisCharacter.getProperty("gender")
            description = thisCharacter.getProperty("description")
            prompt = "portrait of "+gender+", "+description + \
                ", solid white background"+promptSuffix
            portrait = self.doGen(
                prompt, num_inference_steps=self.num_inference_steps)
            portraits[name] = portrait
            yield {"debug": description}
            yield {"image": portrait}

        synopsis = story.getProperty("story synopsis")

        whichScene = whichScene
        numScenes = 3
        for i in range(k):

            scene = str(story.getProperty("scene %d" % whichScene))
            whichSubplot = (i*5//k)+1
            wss = "part %d" % whichSubplot
            thisSubplot = plotOverview.getProperty(wss)
            story.objects['subplot'] = thisSubplot

            audio = self.generate_track_by_prompt_vol(scene, vol=0.25)

            # parse out character1 and character2
            character1 = None
            for this_character1 in [mainCharacter]+supportingCharacters:
                if str(this_character1.getProperty("name")) in scene:
                    character1 = this_character1
                    character1description = character1.getProperty(
                        "description")
                    break
            character2 = None
            for this_character2 in [mainCharacter]+supportingCharacters:
                # gah, bug was that we were finding the same person twice!
                if character1 is not None and str(this_character2.getProperty("name")) == str(character1.getProperty("name")):
                    continue
                if str(this_character2.getProperty("name")) in scene:
                    character2 = this_character2
                    character2description = character2.getProperty(
                        "description")
                    break

            # swap order if needed
            if character1 is not None and character2 is not None:
                name1 = str(character1.getProperty("name"))
                name2 = str(character2.getProperty("name"))
                i1 = scene.index(name1)
                i2 = scene.index(name2)

                #print("indexes", i1, i2)

                if i1 > i2:
                    # swap
                    character1, character2 = character2, character1
                    character1description, character2description = character2description, character1description
            else:
                #print("huh?", character1, character2)
                pass

            prompt = scene + ", "
            if character1 is not None:
                prompt += character1description
            else:
                print("Error, could not find character1", scene)

            if character2 is not None:
                prompt += ", "+character2description+","+promptSuffix
            else:
                print("Error, could not find character2", scene)

            image = self.doGen(
                prompt, num_inference_steps=self.num_inference_steps)

            yield {"debug": "Subplot: %s\n Scene: %s" % (thisSubplot, scene)}
            if audio:
                yield {"music": audio}
            else:
                print("err, no music!")
            yield {"image": image}
            transcript = self.sceneToTranscript(
                story, k=whichScene,
                character1=character1,
                character2=character2,
                whichScene=whichScene,
            )

            if self.verbose:
                print(transcript)

            # generate dialogue
            if skip_transcript == False:
                tt = transcript.getProperty("transcript")
                for line in tt.split("\n"):
                    thisImg = image.copy()
                    name, dialogue = line.split(":")
                    voice = voices[name]
                    portrait = portraits[name]
                    p2 = portrait.resize((portrait_size, portrait_size))
                    thisImg.paste(
                        p2, (thisImg.size[0]-portrait_size, thisImg.size[1]-portrait_size))
                    speech, duration = self.textToSpeech(dialogue, voice)
                    yield {"image": thisImg}
                    yield {"speech": speech,
                           "duration": duration+1,
                           "name": name,
                           "dialogue": dialogue}

            # advance plot if necessary
            whichScene += 1
            if whichScene > numScenes:
                whichScene, numScenes, story = self.advanceStory(
                    story,
                    thisSubplot,
                    mainCharacter=mainCharacter,
                    supportingCharacters=supportingCharacters,
                    alwaysUseMainCharacter=alwaysUseMainCharacter
                )
                if self.verbose:
                    print("advancing scene", story, whichScene, numScenes)
            else:
                #print("not advancing",whichScene,numScenes)
                pass



    def getTagBundles(self,longscreenplay):

        tags=set([x.split(":")[0].lower() for x in longscreenplay.split("\n") if ":" in x])

        tags=[x for x in tags if len(x.split())<4]

        ignored_words=set(["the","name","setting","music","action","sound","effect"])


        tag_bundles=[]

        for tag in tags:
            tagset=set(tag.split())-ignored_words
            if len(tagset)==0:
                continue
            t=0
            for bundle in tag_bundles:
                if tagset.intersection(bundle):
                    t=1
                    bundle.update(tagset)
            if t==0:
                tag_bundles+=[tagset]

        #print(tag_bundles)

        #and let's do that more more time

        new_tag_bundles=[]

        for tagset in tag_bundles:
            t=0
            for bundle in new_tag_bundles:
                if tagset.intersection(bundle):
                    t=1
                    bundle.update(tagset)
            if t==0:
                new_tag_bundles+=[tagset]

        #print(new_tag_bundles)
        
        return new_tag_bundles
    

    def normalizeTag(self,tag,tag_bundles):
        ignored_words=set(["the","name","setting","music","action","sound","effect"])
        tagset=set(tag.split())-ignored_words
        if len(tagset)==0:
            print("this should never happen!")
            return tag
        t=0
        for bundle in tag_bundles:
            if tagset.intersection(bundle):
                return "_".join(bundle)
        print("this should never happen!")
        return tag
    

    def mergeName(self,name1,names):
        ignored_words=set(["the","name","setting","music","action","sound","effect"])
        s1=set(name1.lower().split())-ignored_words
        for name2 in names:
            s2=set(name2.lower().split())-ignored_words
            if s1.intersection(s2):
                return name2
            
        return name1


    def transcriptToAnime(
        self,
        transcript,
        promptSuffix="",
        portrait_size=128,
        aggressiveMerging=False,
        savedcharacters=None,
        savedPortraits=None,
        savedVoices=None,
        savedGenders=None,
        actionDuration=5,
        settingDuration=2
    ):

        # make sure text generator is on cuda (can get out of sync if we ctrl+c during doGen() )
        if self.textGenerator["name"].startswith("GPT3"):
            pass
        else:
            self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cuda()

        
        #extract characters
        if savedcharacters is None:
            _characters={}
        else:
            _characters=savedcharacters
        if savedPortraits is None:
            portraits = {}
        else:
            portraits=savedPortraits
        if savedVoices is None:
            voices = {}
        else:
            voices = savedVoices
        if savedGenders is None:
            genders = {}
        else:
            genders=savedGenders




        tagBundles=self.getTagBundles(transcript)
        for line in transcript.split("\n"):
            tag=line.split(":")[0].strip().lower()
            if tag in ["setting","action","music","sound effect"]:
                continue
            if aggressiveMerging:
                #tagn=self.normalizeTag(tag,tagBundles)
                tagn=self.mergeName(tag,_characters.keys())
            else:
                tagn=tag
            if tagn in _characters:
                continue
            else:
                character=WorldObject(
                    self.templates,
                    self.textGenerator,
                    "character",
                    objects={"name":tag}
                )


                print("GENERATED CHARACTER",character)

                _characters[tagn]=character
                print(tag,character)

        characters=list(_characters.values())



        # get voices
        
        for thisCharacter in characters:
            name = str(thisCharacter.getProperty("name"))            
            gender = thisCharacter.getProperty("gender")


            if name in voices:
                continue

            if gender == "male":
                voices[name] = random.choice(self.maleVoices)
            else:
                voices[name] = random.choice(self.femaleVoices)
            genders[name] = gender
            description = thisCharacter.getProperty("description")

        # generate portraits
        
        for thisCharacter in characters:
            name = str(thisCharacter.getProperty("name"))

            if name in portraits:
                continue

            gender = thisCharacter.getProperty("gender")
            description = thisCharacter.getProperty("description")
            prompt = "portrait of "+gender+", "+description + \
                ", solid white background"+promptSuffix
            portrait = self.doGen(
                prompt, num_inference_steps=self.num_inference_steps)
            portraits[name] = portrait
            yield {"debug": description}
            yield {"image": portrait}
            yield {"caption": "new character: %s"%description,"duration":1}


        
        settingImage = self.doGen(
                "an empty stage", num_inference_steps=self.num_inference_steps)


        for line in transcript.split("\n"):


            if len(line.split(":"))!=2:
                print("this should never happen!",line,transcript)
                continue

            tag=line.split(":")[0].strip().lower()
            description=line.split(":")[1].strip().lower()
            if tag =="setting":
                settingImage = self.doGen(
                    description+promptSuffix, num_inference_steps=self.num_inference_steps)
                
                yield {"image": settingImage}
                yield {"caption": "Setting: %s"%description,
                       "duration":settingDuration}
            
            elif tag=="music":


                audio = self.generate_track_by_prompt_vol(description, vol=0.25)
                yield {"music": audio}

            elif tag=="sound effect":

                #todo: implement 

                yield {"caption": "Sound Effect: %s"%description,
                       "duration":settingDuration}
                

            elif tag=="action":

                actionImage = self.doGen(
                    description+promptSuffix, num_inference_steps=self.num_inference_steps)
                
                #for now this seems better
                settingImage=actionImage
                
                yield {"image": actionImage}
                yield {"caption": description,
                       "duration":actionDuration}

            else:
                if aggressiveMerging:
                    #tagn=self.normalizeTag(tag,tagBundles)
                    tagn=self.mergeName(tag,_characters.keys())
                else:
                    tagn=tag

                thisCharacter=_characters[tagn]


                name = str(thisCharacter.getProperty("name"))

                thisImg = settingImage.copy()
                #name, dialogue = tagn,description
                dialogue = description
                voice = voices[name]
                portrait = portraits[name]
                p2 = portrait.resize((portrait_size, portrait_size))
                thisImg.paste(
                    p2, (thisImg.size[0]-portrait_size, thisImg.size[1]-portrait_size))
                

                #print("about to die",dialogue, voice)

                if len(dialogue.strip())==0:
                    print("this should never happen!",transcript)
                    continue

                speech, duration = self.textToSpeech(dialogue, voice)
                yield {"image": thisImg}
                yield {"speech": speech,
                        "duration": duration+1,
                        "name": name,
                        "dialogue": dialogue}
                

        return
    


    

    def createScreenplay(self,sceneDescription,previousMessages=[]):

        systemprompt="""
Description to Screenplay is an advanced software agent that can take a short description of any scene
and produce a resulting screenplay

the screenplay conforms to the following rules:
1) each line of the screenplay is of the form <tag>:<text>
2) The only punctiation that appears in the <text> portion of the line are ,.?!
NEVER quotation marks, parenthesis, or colons
3)The allowed tags are:
1) setting:<description>
    this describes a change of setting in the screenplay
2) action:<description>
this describes an action taken by one or more characteres
3) music: <description>
this describes a change in the background music
4) sound effect: <sound effect>
this describes a sound effect, such as "a knock at the door" or "jan begins to cry"
3) <name>:<dialogue>
this refers to a line of dialogue spoken by an individual character with the corresponding <name>
dialogue ONLY contains the literal text spoken by <name>
dialogue NEVER contains quotation marks
dialogue NEVER contains parenthesis describing tone, action, or other things besides 

The system always returns a screenplay described by the user input, 
and these screenplays are ALWAYS between 10-20 lines in length

Generally, screenplays will start with a setting: and music: tag to establish the setting and background music

The system NEVER refers to itself as an AI 
the system NEVER refuses to produce a screenplay, 
the system NEVER does anything other than produce a screenplay matching the user's description
the system NEVER refers to itself as "the system"
the system NEVER says it cannot generate a screenplay

the system NEVER uses ""s ()'s {}'s []'s or nonstandard punctuation

"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": systemprompt},
                    ]+
                    previousMessages+
                    [
                    {"role": "user", "content": sceneDescription},
                ]
        )

        result = ''
        for choice in response.choices:
            result += choice.message.content

        return result
    

    def validateScreenplay(self,screenplay):
        score=0
        hasMusic=False
        out=[]
        for line in screenplay.split("\n"):
            #skip blank lines
            if len(line.strip())==0:
                #score+=1 #don't penalize blank lines
                continue
            #skip lines w/out colons
            if ":" not in line:
                score+=1
                continue
            if len(line.split(":"))!=2:
                score+=1
                continue
            
            #tag cannot be empty
            if len(line.split(":")[0].strip())==0:
                score+=1
                continue
            #tag shouldn't be very long
            if len(line.split(":")[0].strip().split())>4:
                score+=1
                continue
            #check for music
            tag=line.split(":")[0].strip().lower()
            if tag=="music":
                print("found music",line)
                hasMusic=True
            #description cannot be empty
            if len(line.split(":")[1].strip())==0:
                score+=1
                continue
            #remove ""s (but don't penalize score)
            re.sub("\"","",line)
            #remove ()'s
            if re.search("\([^\)]*\)",line):
                score+=1
            line=re.sub("\([^\)]*\)","",line)
            #remove ""'s
            line=re.sub("[^a-zA-Z0-9_.?!,';: ]","",line)
            if re.search("[^a-zA-Z0-9_.?!,';: ]",line):
                score+=1
            if len(line.strip())==0:
                score+=1
                continue
            out+=[line]

        #add music if there isn't any
        if hasMusic==False:
            out=["music: %s"%self.riffusionSuffix]+out

        print(out,hasMusic)

        return out, score
    

    def getValidScreenplay(self,sceneDescription,nTrials=3,previousMessages=[],verbose=False):
        bestScreenplay=None
        _bestScreenplay=None
        bestScore=999
        
        for i in range(nTrials):
            s=self.createScreenplay(sceneDescription,previousMessages=previousMessages)
            v,score=self.validateScreenplay(s)

            if verbose:
                print(s,score)

            logging.info("screenplay:\n %s\n score %d/%d",s,score,len(v))
            


            if len(v)>8 and score==0:
                return "\n".join(v)            
            if len(v)>8 and score/len(v)<bestScore:
                bestScore=score
                bestScreenplay=v
                _bestScreenplay=s
            
        if bestScreenplay is None:        
            print("unable to create screenplay!")
            s=self.createScreenplay(sceneDescription,previousMessages=previousMessages)
            print(s)
            v,score=self.validateScreenplay(s)
            return "\n".join(v)        
        else:
            print(_bestScreenplay,bestScore)
            return "\n".join(bestScreenplay)





    def continueSceneGPT(self,novelSummary,characters,chapters,allChapters,whichChapter,whichScene,previousMessages=None,max_tokens=1000,verbose=False):
        
        summarizeNovelMessage=WorldObject(
                self.templates,
                self.textGenerator,
                "explainNovelTemplate",
                objects={"novelSummary":novelSummary,
                        "novelCharacters":characters,
                        "novelChapters":chapters,
                        }
        )
        
        sceneSummary=allChapters[whichChapter-1].getProperty("scene %d summary"%whichScene)
        
        print(summarizeNovelMessage)
        
        print(sceneSummary)
        
        if previousMessages is None:
        
            messages=[
                {"role":"user","content":str(summarizeNovelMessage)}        
            ]+[
                {"role":"user","content":"""

        Create a transcript for chapter {whichChapter}, scene {whichScene} with the following summary

                """.format(whichChapter=whichChapter,whichScene=whichScene,sceneSummary=sceneSummary)
                }
            ]
            
        else:
            
            messages=previousMessages+[
                {"role":"user","content":"""

        Create a transcript for chapter {whichChapter}, scene {whichScene} with the following summary

                """.format(whichChapter=whichChapter,whichScene=whichScene,sceneSummary=sceneSummary)
                }
            ]
            
        
        #response=animeBuilder.createScreenplay(sceneSummary,messages)
        response=self.getValidScreenplay(sceneSummary,previousMessages=messages)
        #response=animeBuilder.getValidScreenplay(sceneSummary)
        
        outputMessages=messages+[
            {"role": "user", "content": sceneSummary},
            {"role": "assistant", "content": response},
        ]
        
        
        return response,outputMessages
    



    def novelToChapters(self,novelSummary,novelCharacters,previousMessages=None,k=12):
        systemPrompt="""
Description to Chapters is an advanced software agent that can take a short description of any novel
and produce a list of chapters.

The list is formatted

chapter 1 title:
<chapter 1 title>
chapter 1 summary:
<chapter 1 title>
chapter 2 title:
<chapter 2 title>
chapter 2 summary:
<chapter 2 title>

With the content in <>'s replaced with appropriate text


the text subsituted for <>'s NEVER contains ":"s
the text subsituted for <>'s is ALWAYS a single line

The system always returns a list of chapters described by the user input, 
and the list of chapters are ALWAYS %d chapters long

The system NEVER refers to itself as an AI 
the system NEVER refuses to produce a list of chapters, 
the system NEVER does anything other than produce a list of chapters matching the user's description
the system NEVER refers to itself as "the system"
the system NEVER says it cannot generate a list of chapters

the system NEVER uses ""s ()'s {}'s []'s or nonstandard punctuation    

"""%k
        
        if previousMessages is None:
            previousMessages=[]

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": systemPrompt},
                ]+
                previousMessages+
                [
                {"role": "user", "content": str(novelCharacters)},
                {"role": "user", "content": str(novelSummary)},
            ]
        )

        result = ''
        for choice in response.choices:
            result += choice.message.content

        return result

    def validateChapters(self,novelChapters,k=12,verbose=False):
        
        #remove blank lines
        customTemplate=""
        for line in novelChapters.split("\n"):
            #drop blank lines
            if len(line.strip())==0:
                continue
            line=line.strip()
            #tags should be lowercase
            if line[-1]==":":
                line=line.lower()
            customTemplate+=line+"\n"
            
            
        if verbose:
            print(customTemplate)
        
        
        
        w=WorldObject(
                self.templates,
                self.textGenerator,
                "novelChapters",
                customTemplate=customTemplate)
        

        
        
        score=0
        for i in range(1,k+1):
            if w.has("chapter %d title"%i) and w.has("chapter %d summary"%i):
                score+=1


        logging.info("%s \n score %d",novelChapters,score)
                
        return w,score

    def getValidChapters(self,novelSummary,characters,k=12,nTrials=3,verbose=False):
        bestNovel=None
        bestScore=0
        for i in range(nTrials):
            c=self.novelToChapters(novelSummary,characters,k=k)
            w,score = self.validateChapters(c,k=k)
            if score==k:
                return w
            if score>bestScore:
                bestNovel=w
                bestScore=score
        print("failed to generate novel",score)
        return w
    


    def chapterToScenes(self,
                        novelSummary,
                        characters,
                        chapters,
                        chapterTitle,
                        chapterSummary,
                        whichChapter,
                        previousMessages=None,k=5):
        systemPrompt="""
    Chapter to Scenes is an advanced software agent that can take a short description of any chapter
    and produce a list of scenes.

    The list is formatted

    scene 1 summary:
    <scene 1 summary>
    scene 2 summary:
    <scene 2 summary>
    scene 3 summary:
    <scene 3 summary>
    scene 4 summary:
    <scene 4 summary>
    scene 5 summary:
    <scene 5 summary>

    With the content in <>'s replaced with appropriate text

    the text subsituted for <>'s NEVER contains ":"s
    the text subsituted for <>'s is ALWAYS a single line
    the text subsituted for <>'s ALWAYS appears on its own line

    The system always returns a list of scenes described by the user input, 
    and the list of scenes are ALWAYS {numScenes} scenes long

    The system NEVER refers to itself as an AI 
    the system NEVER refuses to produce a list of scenes, 
    the system NEVER does anything other than produce a list of scenes matching the user's description
    the system NEVER refers to itself as "the system"
    the system NEVER says it cannot generate a list of scenes

    the system NEVER uses ""s ()'s {{}}'s []'s or nonstandard punctuation    

    The number of scenes produced is {numScenes}, NEVER MORE and NEVER LESS

    """.format(numScenes=k)
        
        
        #print(systemPrompt)
        
        if previousMessages is None:
            messages=[
                {"role": "system", "content": systemPrompt},               
                {"role": "user", "content": str(characters)},
                {"role": "user", "content": str(novelSummary)},
                #{"role": "user", "content": str(chapters)},
                {"role": "user", "content": "generate scenes for chapter %d of this novel"%whichChapter},
                {"role": "user", "content": str(chapterTitle)},
                {"role": "user", "content": str(chapterSummary)},
            ]
        else:
            messages=previousMessages+[
                {"role": "user", "content": "generate scenes for chapter %d of this novel"%whichChapter},
                {"role": "user", "content": str(chapterTitle)},
                {"role": "user", "content": str(chapterSummary)},
            ]
            

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        result = ''
        for choice in response.choices:
            result += choice.message.content
            
        outputMessages=messages+[{"assistant":result}]

        return result,outputMessages


    def validateScenes(self,chapterScenes,k=5,verbose=False):
        
        #remove blank lines
        customTemplate=""
        for line in chapterScenes.split("\n"):
            #drop blank lines
            if len(line.strip())==0:
                continue
            line=line.strip()
            #tags should be lowercase
            if line[-1]==":":
                line=line.lower()
            customTemplate+=line+"\n"
            
            
        if verbose:
            print(customTemplate)
        logging.info(customTemplate)
        
        
        w=WorldObject(
                self.templates,
                self.textGenerator,
                "chapterScenes",
                customTemplate=customTemplate)
        
        score=0
        for i in range(1,k+1):
            if w.has("scene %d summary"%i):
                score+=1
                
        return w,score


    def getValidScenes(
        self,
        novelSummary,
        characters,
        chapters,
        chapterTitle,
        chapterSummary,
        whichChapter,
        k=5,
        nTrials=3,
        previousMessages=None,
        verbose=False
    ):
        bestNovel=None
        bestScore=0
        bestMessages=None
        for i in range(nTrials):
            c,messages=self.chapterToScenes(novelSummary,
                        characters,
                        chapters,
                        chapterTitle,
                        chapterSummary,
                        whichChapter,
                        previousMessages=previousMessages,
                        k=k
                        )
            
            
            w,score = self.validateScenes(c,k=k)
            
            logging.info("scenes:\n %s\n score %d / %d",c,score,k)
            
            if score==k:
                return w,messages
            if score>bestScore:
                bestNovel=w
                bestScore=score
                bestMessages=messages
        print("failed to generate novel",score)
        return bestNovel,bestMessages

    def chaptersToScenes(
        self,
        novelSummary,
        characters,
        chapters,
        numChapters=12,
        numScenes=5,
        nTrials=3
    ):
        
        output=[]
        
        previousMessages=None
        for whichChapter in range(1,numChapters+1):

            chapterTitle=chapters.getProperty("chapter %d title"%(whichChapter))
            chapterSummary=chapters.getProperty("chapter %d summary"%(whichChapter))

            if previousMessages is not None and len(previousMessages)>15:
                previousMessages=previousMessages[:3]+previousMessages[-12:]

            c,messages=self.getValidScenes(
                novelSummary,
                characters,
                chapters,
                chapterTitle,
                chapterSummary,
                whichChapter=1,
                k=numScenes,
                nTrials=nTrials
            )
            
            print("\n\nchapter",whichChapter,chapterTitle,chapterSummary)
            print(c)
            
            output+=[c]
            
            previousMessages=messages
            
        return output
    

    ###these methods are special because they take text templates as inputs instead of
    # WorldObjects

    def create_novel_summary(self,story_objects):

        storyObjects=WorldObject(
            self.templates,
            self.textGenerator,
            "storyObjects",
            customTemplate=story_objects
        )

        novelSummary=WorldObject(
            self.templates,
            self.textGenerator,
            "novelSummary",
            #objects={"title":"The big one",
            #        "summary":"A group of sexy female lifeguards take up surfing"}
            objects={"storyObjects":storyObjects}
        )

        novel_summary = str(novelSummary)
        return novel_summary.split('\n', 1)[1]



    def create_characters(self,story_objects,novel_summary):

        storyObjects=WorldObject(
            self.templates,
            self.textGenerator,
            "storyObjects",
            customTemplate=story_objects
        )

        novelSummary=WorldObject(
            self.templates,
            self.textGenerator,
            "novelSummary",
            customTemplate=novel_summary
        )

        characters=WorldObject(
            self.templates,
            self.textGenerator,
            "novelCharacters",
            objects={"novelSummary":novelSummary,
                     "storyObjects":storyObjects
                     },
        )

        return str(characters).split('\n', 1)[1]
        #return str(characters)
    

    def create_chapters(self,story_objects,novel_summary, _characters, num_chapters,nTrials=3):

        storyObjects=WorldObject(
            self.templates,
            self.textGenerator,
            "storyObjects",
            customTemplate=story_objects
        )

        novelSummary=WorldObject(
                self.templates,
                self.textGenerator,
                "novelSummary",
                customTemplate=novel_summary
            )
        
        characters=WorldObject(
                self.templates,
                self.textGenerator,
                "novelCharacters",
                customTemplate=_characters
            )
        
        chapters=self.getValidChapters(
            novelSummary,
            characters,
            k=num_chapters,
            nTrials=nTrials
        )

        return str(chapters).split('\n', 1)[1]
        

    def create_scenes(self,story_objects,novel_summary, _characters, _chapters, num_chapters, num_scenes,nTrials=3):

        novelSummary=WorldObject(
                self.templates,
                self.textGenerator,
                "novelSummary",
                customTemplate=novel_summary
            )
        
        characters=WorldObject(
                self.templates,
                self.textGenerator,
                "novelCharacters",
                customTemplate=_characters
            )
        
        chapters=WorldObject(
                self.templates,
                self.textGenerator,
                "chapters",
                customTemplate=_chapters
            )
        
        scenes=self.chaptersToScenes(
            novelSummary,
            characters,
            chapters,
            numChapters=num_chapters,
            numScenes=num_scenes,
            nTrials=nTrials
        )

        return "\n===\n".join([str(x).split('\n', 1)[1] for x in scenes])


    def generate_movie_data(self,story_objects,novel_summary, _characters, _chapters, scenes, num_chapters, num_scenes):
        # Process the inputs and generate the movie data
        # This is where you would include your existing code to generate the movie elements
        # For demonstration purposes, we'll just yield some dummy elements

        storyObjects=WorldObject(
            self.templates,
            self.textGenerator,
            "storyObjects",
            customTemplate=story_objects
        )

        #convert back into correct format
        novelSummary=WorldObject(
                self.templates,
                self.textGenerator,
                "novelSummary",
                customTemplate=novel_summary
            )
        
        characters=WorldObject(
                self.templates,
                self.textGenerator,
                "novelCharacters",
                customTemplate=_characters
            )
        
        chapters=WorldObject(
                self.templates,
                self.textGenerator,
                "chapters",
                customTemplate=_chapters
            )
        
        chapters=WorldObject(
                self.templates,
                self.textGenerator,
                "chapters",
                customTemplate=_chapters
            )
        
        all_scenes=scenes.split("===")

        allScenes=[
            WorldObject(
                self.templates,
                self.textGenerator,
                "chapterScenes",
                customTemplate=_scenes
            )

            for _scenes in all_scenes
        ]

        mainCharacter=WorldObject(
                    self.templates,
                    self.textGenerator,
                    "character",
                    objects={
                        "name":characters.getProperty("main character name"),
                        "description text":characters.getProperty("main character description"),
                            },
        )

        supportingCharacter1=WorldObject(
                    self.templates,
                    self.textGenerator,
                    "character",
                    objects={
                        "name":characters.getProperty("supporting character 1 name"),
                        "description text":characters.getProperty("supporting character 1 description"),
                            }
        )

        supportingCharacter2=WorldObject(
                    self.templates,
                    self.textGenerator,
                    "character",
                    objects={
                        "name":characters.getProperty("supporting character 2 name"),
                        "description text":characters.getProperty("supporting character 2 description"),
                            }
        )

        antagonist=WorldObject(
                    self.templates,
                    self.textGenerator,
                    "character",
                    objects={
                        "name":characters.getProperty("antagonist name"),
                        "description text":characters.getProperty("antagonist description"),
                            },
            #verbose=True
        )

        savedcharacters={
            str(mainCharacter.getProperty("name")):mainCharacter,
            str(supportingCharacter1.getProperty("name")):supportingCharacter1,
            str(supportingCharacter2.getProperty("name")):supportingCharacter2,
            str(antagonist.getProperty("name")):antagonist,
        }
        savedPortraits={}
        savedVoices={}
        savedGenders={}

        previousScene=None

        for whichChapter in range(1,num_chapters+1):
            for whichScene in range(1,num_scenes+1):


                yield {"debug":"chapter %d scene %d"%(whichChapter,whichScene)}

                if previousScene is not None:
                    previousMessages=previousScene[1]
                    
                    
                    #trim messages when n>3 1+3*n=10
                    
                    if len(previousMessages)>10:
                        previousMessages=previousMessages[:1]+previousMessages[-9:]
                    
                    #thisScene=continueSceneGPT(whichChapter,whichScene,previousMessages)
                    thisScene=self.continueSceneGPT(novelSummary,characters,chapters,allScenes,whichChapter,whichScene,previousMessages)
                    
                else:
                    
                    #thisScene=continueSceneGPT(whichChapter,whichScene)
                    thisScene=self.continueSceneGPT(novelSummary,characters,chapters,allScenes,whichChapter,whichScene)
                    
                s=thisScene[0]

                if novelSummary.has("characterType"):
                    promptSuffix=", "+novelSummary.getProperty("characterType")+self.suffix
                else:
                    promptSuffix=self.suffix
                    
                    
                anime=self.transcriptToAnime(
                    s,
                    portrait_size=256,
                    promptSuffix=promptSuffix,
                    savedcharacters=savedcharacters,
                    savedPortraits=savedPortraits,
                    savedVoices=savedVoices,
                    savedGenders=savedGenders,
                    aggressiveMerging=True,
                )
                for storyElement in anime:
                    yield storyElement
                
                previousScene=thisScene

        yield {"caption": "THE END",
                       "duration":1}







