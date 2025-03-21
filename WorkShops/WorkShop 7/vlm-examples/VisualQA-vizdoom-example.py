############################################################################
# Program extended from https://huggingface.co/Salesforce/blip-vqa-base
#
# This small program illustrates that the BLIP vision language model below
# can play the game of VizDoom -- at least to some extent. You should try
# to integrate this functionality into the code of last week's workshop.
# First, as part of file bc_VizDoom_FromDemonstration.py for illustration
# purposes, and then as part of sb_VizDoom.py.
# 
# Contact: hcuayahuitl@lincoln.ac.uk
# Last update on 19 March 2025.
############################################################################

import os
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# Load model and processor for visual question answering
model_id = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_id)
model_vaq = BlipForQuestionAnswering.from_pretrained(model_id)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_vaq.to(device)

# Generate agent moves from a folder containing example images
folder_path = "./some-vizdooom-images"
for file_name in os.listdir(folder_path):
    image_path = folder_path+'/'+file_name  
    image = Image.open(image_path).convert("RGB")

    # obtain a binary matching value
    prompt = f"Should i move left or right to avoid being hit by a fire ball?"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad(): output = model_vaq.generate(**inputs)
    action = processor.decode(output[0], skip_special_tokens=True)
    print(f"File: {image_path} Question: {prompt} Response: {action}")
