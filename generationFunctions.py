from diffusers import StableDiffusionImg2ImgPipeline
import torch
import gc
import random

class GenerationFunctions:
    def __init__(self,saveMemory=False):

        self.saveMemory=saveMemory

        print("LOADING IMG2IMG MODEL")                                                              
        
        self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained('Deci/DeciDiffusion-v1-0',
                                                        custom_pipeline='D:/img/DeciDiffusion-v1-0',
                                                        torch_dtype=torch.float16,
                                                        safety_checker=None,
                                                        )

        self.img2img.unet = self.img2img.unet.from_pretrained('Deci/DeciDiffusion-v1-0',
                                                    subfolder='flexible_unet',
                                                    torch_dtype=torch.float16)
        

        #move to GPU (unless saveMemory is True)
        if not self.saveMemory:
            self.img2img = self.img2img.to('cuda')
        
    def image_to_image(self, image,prompt,n_prompt,strength=0.25,seed=-1,steps=30):

        if self.saveMemory:
            # Move pipeline to device
            self.img2img = self.img2img.to('cuda')
        
        if seed==-1:
            seed=random.randint(0,100000)

        negative_prompt="low resolution, blurry, "+n_prompt

        
        # Call the pipeline function directly
        result = self.img2img(prompt=[prompt],
                        negative_prompt=[n_prompt],
                        image=image,
                        strength=strength,
                        generator=torch.Generator("cuda").manual_seed(seed),
                        num_inference_steps=steps)

        img = result.images[0]

        if self.saveMemory:
            # Move pipeline back to CPU
            self.img2img = self.img2img.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()


        return img
    

        