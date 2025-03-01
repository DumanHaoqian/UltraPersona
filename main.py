##############################################
#            Persona Parameter Tester        #
#       Exhaustive Combination Testing       #
#                IN POLYU                    #
#            Author: DU Haoqian              #
#                2025/2/28                   #
##############################################

import os
import cv2
import torch
import traceback
from persona import Persona  
INPUT_IMAGE_PATH = "/home/haoqian/PI/ultraPersona/test_dhq.jpeg"
BASE_PROMPT = "A beautiful portrait"
OUTPUT_ROOT = "/home/haoqian/PI/ultraPersona/Test_Results"  

BACKGROUND_CHOICES = range(1, 9)  
STYLE_CHOICES = range(1, 10)      #

def initialize_test_environment():

    for style in STYLE_CHOICES:
        style_dir = os.path.join(OUTPUT_ROOT, f"style_{style}")
        os.makedirs(style_dir, exist_ok=True)
    
    input_image = cv2.imread(INPUT_IMAGE_PATH)
    if input_image is None:
        raise FileNotFoundError(f"Input image not found at {INPUT_IMAGE_PATH}")
    return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

def generate_filename(background, style, step):

    return f"bg_{background}_style_{style}_step{step}.png"

def test_single_combination(background, style, input_image):

    try:
        print(f"\n=== Testing combination: BG={background}, Style={style} ===")
        

        persona = Persona(
            background_choice=background,
            input_image=input_image,
            styling_choice=style,
            prompt=BASE_PROMPT
        )


        print("Running face generation...")
        step1_img = persona.face_generator_emb()
        cv2.imwrite(os.path.join(OUTPUT_ROOT, f"style_{style}", generate_filename(background, style, 1)), step1_img)

        print("Running background merging...")
        step2_img = persona.extract_and_merge()
        cv2.imwrite(os.path.join(OUTPUT_ROOT, f"style_{style}", generate_filename(background, style, 2)), step2_img)

        print("Running style transfer...")
        final_img = persona.img_styling()
        cv2.imwrite(os.path.join(OUTPUT_ROOT, f"style_{style}", generate_filename(background, style, 3)), final_img)


        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"!! Error in BG={background}, Style={style}: {str(e)}")
        traceback.print_exc()
        return False

def main():

    input_image = initialize_test_environment()
    success_count = 0
    total_combinations = len(BACKGROUND_CHOICES) * len(STYLE_CHOICES)


    for bg in BACKGROUND_CHOICES:
        for style in STYLE_CHOICES:
            if test_single_combination(bg, style, input_image):
                success_count += 1


    print(f"\nTest completed: {success_count}/{total_combinations} successful")
    print(f"Failure rate: {(total_combinations-success_count)/total_combinations:.1%}")
    print(f"Results saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()