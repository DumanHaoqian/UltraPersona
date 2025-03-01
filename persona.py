##############################################
#             PolySmart Persona              #
#       Personalized Image Generator         #
# Generate the Images with multiple stylings #
#                IN POLYU                    #
#            Author: DU Haoqian              #
#                2025/2/28                   #
##############################################
import numpy as np
import cv2
import PIL.Image as Image
import torch
import random
from ip_adaptor_face import step1_generate_img_face as step1
from unet_merge import step2_unet as step2
from ImageStyling.protovision import protovision
from ImageStyling.Onelastkiss import one_last_kiss
from ImageStyling.animation import animation
from ImageStyling.dreamshaper import dreamshaper
from ImageStyling.pix2pix import pix2pix

class Persona:
    def __init__( self, background_choice,input_image: np.ndarray,styling_choice, prompt: str):
        self.background_choice=background_choice
        '''
        The Backgrounds are:
        1: "Jockey Club Innovation Tower",
        2: "P504 Student Lab",
        3: "PolyU Great Lawn",
        4: "A Garden",
        5: "Clock Square",
        6: "Library Gate",
        7: "Library G Floor",
        8: "PolyU Garden Restaurant"
        '''
        self.styling_choice=styling_choice
        '''
        Ths stylings are:
        1: "Van Gogh Starry Night", 
        2: "One last kiss",
        3: "Professional 3d",
        4: "JoJo Style",
        5: "Cyberpunk 2077",
        6: "Medieval Style",
        7: "Abstract Animation",
        8: "Miro Style",
        9: "Modigliani"
        '''
        self.prompt=prompt
        self.input_image=input_image
        self.step1_intermedia_img=None
        self.step2_merged=None
        self.final_img=None
        self.storage_mapping={
            1:"/home/haoqian/PI/ultraPersona/Result_imgs/Starry_night",
            2:"/home/haoqian/PI/ultraPersona/Result_imgs/One_last_kiss",
            3:"/home/haoqian/PI/ultraPersona/Result_imgs/Professional_3D",
            4:"/home/haoqian/PI/ultraPersona/Result_imgs/JoJo_style",
            5:"/home/haoqian/PI/ultraPersona/Result_imgs/CyberPunk2077",
            6:"/home/haoqian/PI/ultraPersona/Result_imgs/Medieval",
            7:"/home/haoqian/PI/ultraPersona/Result_imgs/Abstract_Animation",
            8:"/home/haoqian/PI/ultraPersona/Result_imgs/Miro",
            9:"/home/haoqian/PI/ultraPersona/Result_imgs/Modigliani"
        }

    def face_generator_emb(self):
        '''Step I IP-Adaptor-Face-id of Persona: To get a high resolution and clean Image(Easy to segment)
            generate_image_from_face()
            Generate an image from a face in the given CV2 image array using a prompt.
            :param image: Input image containing a face as a NumPy array (CV2 format).
            :param prompt: Text prompt for image generation.
            :param seed: Random seed for reproducibility.
            :return: Generated image as a NumPy array.
            '''
        step1_face_img=step1.generate_image_from_face(self.input_image,self.prompt)
        self.step1_intermedia_img=step1_face_img
        return step1_face_img
    
    def extract_and_merge(self):
        '''Step II UNET Segmentation and Merging:'''
        merged_img_with_background=step2.extract_and_place_person(self.step1_intermedia_img,self.background_choice)
        self.step2_merged=merged_img_with_background
        return merged_img_with_background

    def feather_edges(image, feather_radius=20):

        alpha = image.split()[-1]  
        alpha = np.array(alpha)

        mask = np.zeros_like(alpha, dtype=np.float32)
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                dist_to_edge = min(x, w - x, y, h - y)
                if dist_to_edge < feather_radius:
                    mask[y, x] = dist_to_edge / feather_radius
                else:
                    mask[y, x] = 1.0

        alpha = (alpha * mask).astype(np.uint8)
        image.putalpha(Image.fromarray(alpha))
        return image

    def img_styling(self):

        if (self.styling_choice==1) :
            # protovision.generate() will return PIL.Image
            Augmented_prompt="Van Gogh Starry Night , swirling, vibrant, impasto, expressionism, color contrast, emotional intensity, night sky, stars, cypress trees, brushwork, post-impressionism, dreamlike, atmospheric"
            pil_result=protovision.generate(self.step2_merged, Augmented_prompt)
            self.final_img=np.array(pil_result)

        if (self.styling_choice==2) :
            pil_result=one_last_kiss.one_last_kiss(self.step2_merged)
            self.final_img=np.array(pil_result)

        if (self.styling_choice==3) :
            # protovision.generate() will return PIL.Image
            Augmented_prompt="Professional 3d cartoon portrait of a man, pixar art style character, octane render, highly detailed"
            pil_result=protovision.generate(self.step2_merged, Augmented_prompt)
            self.final_img=np.array(pil_result)

        if (self.styling_choice==4):
            Augmented_prompt="JoJo Style, JoJo, absurdres, anime style, vibrant colors,detailed character design, high-quality illustration, masterpiece,stunning visuals, dynamic composition, cinematic lighting, emotional expression, intricate backgrounds"
            pil_result=animation.generate(self.step2_merged,Augmented_prompt)
            self.final_img=np.array(pil_result)

        if (self.styling_choice==5):
            Augmented_prompt="wearing cyberpunk 2077 style, solo character in a vibrant cyberpunk cityscape with flying cars in the sky, neon lights illuminating the streets, futuristic skyscrapers, holographic advertisements, bustling crowds, rain-soaked pavement reflecting vibrant colors, realistic portrayal, captivating individual, (highly detailed face), slender build, attractive features, 8K resolution, expressive lips, high definition, HDR, photorealistic style, ultra high resolution, hyper realism, open mouth, subtle facial hair"
            pil_result=dreamshaper.generate(self.step2_merged, Augmented_prompt)
            self.final_img=np.array(pil_result)

        if (self.styling_choice==6):
            Augmented_prompt="wearing medieval architectural style, solo character, realistic portrayal, captivating individual, (highly detailed face), slender build, attractive features, 8K resolution, expressive lips, high definition, HDR, photorealistic style, ultra high resolution, hyper realism, open mouth, subtle facial hair"
            pil_result=dreamshaper.generate(self.step2_merged,Augmented_prompt)
            self.final_img=np.array(pil_result)

        if (self.styling_choice==7):
            Augmented_prompt="Make it like a animation"
            pil_result=pix2pix.generate(self.step2_merged,Augmented_prompt)
            self.final_img=np.array(pil_result) 

        if (self.styling_choice==8):
            Augmented_prompt="Make it a Miro painting"
            pil_result=pix2pix.generate(self.step2_merged,Augmented_prompt)
            self.final_img=np.array(pil_result
                                    )  
        
        if (self.styling_choice==9):
            Augmented_prompt="Make it a Modigliani painting"
            pil_result=pix2pix.generate(self.step2_merged,Augmented_prompt)
            self.final_img=np.array(pil_result) 
        
        if (self.background_choice==2 and (self.styling_choice!=2 and self.styling_choice!=7 and self.styling_choice!=8 and self.styling_choice!=9 ) ):
            background=Image.fromarray(self.final_img)
            insert_image_path="/home/haoqian/PI/ultraPersona/logo/logo_lab.png"
            insert_image = Image.open(insert_image_path)
            insert_image=np.array(insert_image)
            insert_image=cv2.cvtColor(insert_image,cv2.COLOR_BGRA2RGBA)
            insert_image=Image.fromarray(insert_image)
            insert_position = (370, 40)
            scale_factor = 1.2  
            new_size = (int(insert_image.width * scale_factor), int(insert_image.height * scale_factor))
            insert_image = insert_image.resize(new_size, Image.LANCZOS)
            background.paste(insert_image, insert_position, insert_image)
            self.final_img=np.array(background)

        if self.background_choice==6 and (self.styling_choice!=2 and self.styling_choice!=7 and self.styling_choice!=8 and self.styling_choice!=9 ):
            background=Image.fromarray(self.final_img)
            insert_image_path="/home/haoqian/PI/ultraPersona/logo/logo_lib.png"
            insert_image = Image.open(insert_image_path)
            insert_image=np.array(insert_image)
            insert_image=cv2.cvtColor(insert_image,cv2.COLOR_BGRA2RGBA)
            insert_image=Image.fromarray(insert_image)
            insert_position = (990, 487)
            scale_factor = 1.5  
            new_size = (int(insert_image.width * scale_factor), int(insert_image.height * scale_factor))
            insert_image = insert_image.resize(new_size, Image.LANCZOS)
            background.paste(insert_image, insert_position, insert_image)
            self.final_img=np.array(background)
        
        out_path=self.storage_mapping[self.styling_choice]
        
        cv2.imwrite(f"{out_path}/{random.randint(0,100000)}.png",self.final_img)
        return self.final_img


if __name__ == "__main__":
 
#Sample Usage:
    #input_image = cv2.imread(r"path/to/your/image")
    input_image = cv2.imread(r"/home/haoqian/PI/demo_imgs/people/1.jpeg")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    psa = Persona(
        background_choice= 1, # [1,2,3,4,5,6,7,8]
        input_image=input_image, 
        styling_choice=1,  # [1,2,3,4,5,6,7,8,9]
        prompt="Your prompt here"
    )

    psa.face_generator_emb()
    psa.extract_and_merge()
    torch.cuda.empty_cache()
    final_img=psa.img_styling()
    cv2.imwrite("final_image.jpg", final_img)

    '''    
    input_image = cv2.imread(r"/home/haoqian/PI/demo_imgs/people/1.jpeg")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    psa = Persona(
        background_choice=6,
        input_image=input_image,
        styling_choice=2,  
        prompt="A beautiful portrait"
    )
    
    test1 = psa.face_generator_emb()
    cv2.imwrite("test_step1_image.jpg", test1)
    
    test2 = psa.extract_and_merge()
    cv2.imwrite("test_step2_image.jpg", test2)
    
    torch.cuda.empty_cache()
    
    test3 = psa.img_styling()
    cv2.imwrite("test_step3_image.jpg", test3)
'''

