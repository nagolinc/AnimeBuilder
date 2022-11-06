import re
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import random
from PIL import Image
import urllib
from pydub import AudioSegment
from io import BytesIO
import numpy as np
from IPython.display import Audio, display
import ipywidgets as widgets
from torch import autocast
import time
from diffusers import StableDiffusionPipeline
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

from mubert import generate_track_by_prompt
from templates import templates
from worldObject import WorldObject


# diffusionModel="hakurei/waifu-diffusion"
# diffusionModel="nitrosocke/mo-di-diffusion"
#diffusionModel = "E:\img\StableCraft\wdf"
# diffusionModel="DGSpitzer/Cyberpunk-Anime-Diffusion"

class AnimeBuilder:

    def __init__(
        self,
        textModel='EleutherAI/gpt-neo-1.3B',
        diffusionModel="hakurei/waifu-diffusion",
        templates=templates,
        advanceSceneObjects=None,
        no_repeat_ngram_size=8,
        repetition_penalty=2.0,
        num_inference_steps=30,
        MIN_ABC=2,
        num_beams=8,
        temperature=1.0,
        verbose=False
    ):

        self.verbose = verbose

        self.templates = templates

        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty

        self.num_inference_steps = num_inference_steps
        self.num_beams = num_beams
        self.temperature = temperature
        self.MIN_ABC = MIN_ABC


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

        print("LOADING TEXT MODEL")

        # text model
        self.textModel = textModel

        textGenerator = pipeline('text-generation',
                                 torch_dtype=torch.float16,
                                 model=self.textModel, device=0)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.textModel, torch_dtype=torch.float16)
        # self.textModel = AutoModelForCausalLM.from_pretrained(
        #    self.textModel, torch_dtype=torch.float16).to('cuda')

        self.textGenerator = {
            'tokenizer': self.tokenizer,
            # 'model': self.textModel
            'pipeline': textGenerator
        }

        # image model

        print("LOADING IMAGE MODEL")

        # make sure you're logged in with `huggingface-cli login`
        self.pipe = StableDiffusionPipeline.from_pretrained(
            diffusionModel,
            # vae=vae,
            # revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True,
            cache_dir="./AI/StableDiffusion")

        self.pipe.safety_checker = None

        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()

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

    def doGen(self, prompt, num_inference_steps=30):
        image = self.pipe([prompt],
                          negative_prompt=[
                              "text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body"],
                          guidance_scale=7.5,
                          num_inference_steps=num_inference_steps,
                          height=512).images[0]
        return image

    def textToSpeech(self, text, voice):
        try:
            with autocast("cuda"):
                self.tts_task.data_cfg.hub["speaker"] = voice
                sample = TTSHubInterface.get_model_input(self.tts_task, text)
                #print("about to die",models[0],sample)
                wav, rate = TTSHubInterface.get_prediction(
                    self.tts_task, self.tts_models[0], self.tts_generator, sample)
                # print("huh?",wav,rate,len(wav)/rate)
                duration = len(wav)/rate
            return (ipd.Audio(wav.cpu(), rate=rate, autoplay=True)), duration
        except:
            print("Error generating text", text, voice)
    # music

    def generate_track_by_prompt_vol(self, prompt, vol=1.0, duration=30, loop=True, autoplay=True):
        url = generate_track_by_prompt(prompt, duration, loop)
        if url is None:
            return
        mp3 = urllib.request.urlopen(url).read()
        original = AudioSegment.from_mp3(BytesIO(mp3))
        samples = original.get_array_of_samples()
        samples /= np.max(np.abs(samples))
        samples *= vol
        audio = Audio(samples, normalize=False,
                      rate=original.frame_rate, autoplay=autoplay)
        return audio

    def descriptionToCharacter(self, description):
        thisObject = WorldObject(self.templates, self.textGenerator, "descriptionToCharacter", objects={
            "description": description},
            genTextAmount_min=self.amtMin,
            genTextAmount_max=self.amtMax,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
            num_beams=self.num_beams,
            temperature=self.temperature,
            MIN_ABC=self.MIN_ABC,
            verbose=self.verbose
        )
        return thisObject

    def advanceStory(self, story, subplot, mainCharacter=None, supportingCharacters=None):

        advanceSceneObject = random.choice(self.advanceSceneObjects)

        # update subplot

        character1 = mainCharacter
        character2 = character2 = random.choice(supportingCharacters)

        if character1 is None:
            character1 = story.getProperty("character1")
        if character2 is None:
            character2 = story.getProperty("character2")
        newStory = WorldObject(self.templates, self.textGenerator, advanceSceneObject['object'], objects={
            "character1": character1,
            "character2": character2,
            "story synopsis": story.getProperty("story synopsis"),
            "subplot": story.getProperty("subplot"),
            "scene 1 previous": story.getProperty("scene 1"),
            "scene 2 previous": story.getProperty("scene 2"),
            "scene 3 previous": story.getProperty("scene 3"),
        },
            genTextAmount_min=self.amtMin,
            genTextAmount_max=self.amtMax,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
            num_beams=self.num_beams,
            temperature=self.temperature,
            MIN_ABC=self.MIN_ABC,
            verbose=self.verbose
        )

        whichScene = advanceSceneObject['whichScene']
        numScenes = advanceSceneObject['numScenes']

        return whichScene, numScenes, newStory

    def sceneToTranscript(self, scene, k=3, character1=None, character2=None):
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
        thisObject = WorldObject(self.templates, self.textGenerator,
                                 "sceneToTranscript", objects,
                                 genTextAmount_min=self.amtMin,
                                 genTextAmount_max=self.amtMax,
                                 no_repeat_ngram_size=self.no_repeat_ngram_size,
                                 repetition_penalty=self.repetition_penalty,
                                 num_beams=self.num_beams,
                                 temperature=self.temperature,
                                 MIN_ABC=self.MIN_ABC,
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
        whichScene=1  # optionally skip first few scenes
    ):

        self.amtMin=amtMin
        self.amtMax=amtMax

        objects = {}
        if synopsis:
            objects['story synopsis'] = synopsis
        if scene1:
            objects['scene 1 text'] = scene1
        if character1:
            if isinstance(character1, str):
                character1 = self.descriptionToCharacter(character1)
                # print(character1)
            objects['character1'] = character1

        if subplot1:
            objects['part 1'] = subplot1

        plotOverview = WorldObject(
            self.templates, self.textGenerator, "plot overview",
            genTextAmount_min=self.amtMin,
            genTextAmount_max=self.amtMax,
            objects=objects,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
            num_beams=self.num_beams,
            MIN_ABC=self.MIN_ABC,
            verbose=self.verbose
        )

        subplot = plotOverview.getProperty("part 1")
        objects["subplot"] = subplot

        story = WorldObject(self.templates, self.textGenerator,
                            "storyWithCharacters",
                            genTextAmount_min=self.amtMin,
                            genTextAmount_max=self.amtMax,
                            objects=objects,
                            no_repeat_ngram_size=self.no_repeat_ngram_size,
                            repetition_penalty=self.repetition_penalty,
                            num_beams=self.num_beams,
                            temperature=self.temperature,
                            MIN_ABC=self.MIN_ABC,
                            verbose=self.verbose
                            )

        # have to do scene 1 and scene 2
        character = story.getProperty("character1")

        if isinstance(character, str):
            characterDescription = character
        else:
            characterDescription = character.getProperty("description")

        supportingCharacters = [story.getProperty(
            "character%d" % i) for i in range(2, 5)]

        if num_characters > 4:
            for i in range(4, num_characters):
                newCharacter = WorldObject(self.templates, self.textGenerator, "character",
                                           genTextAmount_min=self.amtMin,
                                           genTextAmount_max=self.amtMax,
                                           no_repeat_ngram_size=self.no_repeat_ngram_size,
                                           repetition_penalty=self.repetition_penalty,
                                           num_beams=self.num_beams,
                                           temperature=self.temperature,
                                           MIN_ABC=self.MIN_ABC,
                                           verbose=self.verbose
                                           )
                print(newCharacter.__repr__())
                supportingCharacters += [newCharacter]

        print(story)

        # get voices
        voices = {}
        genders = {}
        for thisCharacter in [character]+supportingCharacters:
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
        for thisCharacter in [character]+supportingCharacters:
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

            character2 = None
            for this_character2 in supportingCharacters:
                if str(this_character2.getProperty("name")) in scene:
                    character2 = this_character2
                    character2description = character2.getProperty(
                        "description")
                    break

            prompt = scene + ", "+characterDescription
            if character2 is not None:
                prompt += ", "+character2description+","+promptSuffix
            image = self.doGen(
                prompt, num_inference_steps=self.num_inference_steps)

            yield {"debug": "Subplot: %s\n Scene: %s" % (thisSubplot, scene)}
            if audio:
                yield {"music": audio}
            else:
                print("err, no music!")
            yield {"image": image}
            transcript = self.sceneToTranscript(
                story, k=whichScene, character2=character2)

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
                    thisImg.paste(p2, (512-portrait_size, 512-portrait_size))
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
                    mainCharacter=character,
                    supportingCharacters=supportingCharacters
                )
                print("advancing scene", story, whichScene, numScenes)
            else:
                #print("not advancing",whichScene,numScenes)
                pass
