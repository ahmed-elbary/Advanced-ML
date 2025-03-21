############################################################################
# Program extended from https://huggingface.co/Salesforce/blip-vqa-base
#
# It illustrates the use of a pre-trained Visual Language Model (VLM) for 
# the task of Visual Question Answering. Since the capability of sentence
# undestanding may be limited in VLMs, two ways of answering questions are
# illustrated below. The first one is based on matching a question and 
# answer pair, and the second is via the calculation of sentence similarity
# scores between the grouth trugh answer and the predicted one. 
# 
# Possible dependencies depending on your PC and previous installations:
# pip install transformers
# pip install sentence-transformers
# 
# Once you have decided on the most useful way of answering questions, you  
# could integrate this functionality into the code of last week's workshop.
# to integrate this functionality into the code of workshop 5 (transformers).
#
# In particular, as part of file ITM_Classifier-baselines.py
# 
# Paper describing the model used below: https://arxiv.org/pdf/2201.12086
# It comes in two versions:
# (1) "Salesforce/blip-vqa-base" uses a ViT-base encoder and small BERT text decoder
# (2) "blip-vqa-capfilt-large" uses a ViT-large encoder and large BERT text decoder
# 
# An alternative to the above is "Salesforce/blip2-opt-2.7b", uses a frozen image
# encoder and a frozen LLM decoder for training a lightweight Query Transformer
# (text representations of images) to connect vision and language.
# Link: https://proceedings.mlr.press/v202/li23q/li23q.pdf
#
# An even more recent model to the two above is known as xGen-MM (BLIP-3).
# Link: https://www.salesforceairesearch.com/opensource/xGen-MM/index.html
# 
# Contact: hcuayahuitl@lincoln.ac.uk
# Last update on 19 March 2025.
############################################################################

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# Load model and processor for visual question answering
model_id = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_id)
model_vaq = BlipForQuestionAnswering.from_pretrained(model_id)

# Load model for sentence comparison/similarity
model_ss = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_vaq.to(device)

# Load an image
image_path = "./some-visual7w-images/v7w_107912.jpg"  # test image
image = Image.open(image_path).convert("RGB")

# Define the question
question = "Where was the picture taken?"
options = ['At the beach.', 'On a lake.', 'On a boat.', 'At the ocean.']

# Preprocess image and question
inputs = processor(images=image, text=question, return_tensors="pt").to(device)

# Generate answer
with torch.no_grad():
    output = model_vaq.generate(**inputs)

# Decode answer
answer = processor.decode(output[0], skip_special_tokens=True)
answer_embedding = model_ss.encode(answer, convert_to_tensor=True)

# Print results of two ways of answering questions
print(f"Image: {image_path}")
print(f"Question: {question}")
for option in options:
    # (1) Obtain a binary matching value
    prompt = f"Do the following question and answer match, yes or no? Question={question}, Answer={option}"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad(): output = model_vaq.generate(**inputs)
    match = processor.decode(output[0], skip_special_tokens=True)

    # (2) Obtain a matching score based on the similarity between ground truth option and predicted answer
    #     The answer with the highest score is arguably the best answer for the given question.
    option_embedding = model_ss.encode(option, convert_to_tensor=True)
    score = util.pytorch_cos_sim(answer_embedding, option_embedding).cpu().item()
    print(f"Option: {option} Predicted Answer: {answer} Match: {match} Similarity_Score: {score}")
