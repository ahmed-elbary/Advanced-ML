############################################################################
# This program is a test from a llama model available from: 
# https://huggingface.co/meta-llama/Llama-3.2-1B
#
# You will have to use your own credentials to run this program:
# pip install huggingface_hub
# huggingface-cli login
############################################################################


import time
from transformers import pipeline

# Define model ID for Llama 3.2-1B
model_id = "meta-llama/Llama-3.2-1B"

# Load the model using the pipeline
pipe = pipeline("text-generation", model=model_id, device=0)  # Use GPU (0 for the first GPU)

# Generate responses to your own prompts
while True:
    prompt = input("How can I help you? ")
    start_time = time.time()
    result = pipe(prompt)#"low insulin to glucagon ratio is seen in all of these except, select one of the following: (A) glycogen synthesis, (B) glycogen breakdown, (C) gluconeogenesis, or (D) ketogenesis.")
    proc_time = start_time - time.time()
    print("Response=%s (%s seconds)" % (result, proc_time))

####################################
# EXAMPLE CHAT with Llama 3.2-1B: 
#
# Device set to use cuda:0
# How can I help you? What is a language model? 
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# Response=[{'generated_text': 'What is a language model? This is a question that has been asked for a long time. The answer is not always clear,'}] (-0.4788551330566406 seconds)
# How can I help you? what are the main swimming styles and which one is the hardest?
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# Response=[{'generated_text': 'what are the main swimming styles and which one is the hardest??\nWhat are the most popular swimming styles?\nWhat is the most popular swimming style?\nWhat is the'}] (-0.5678412914276123 seconds)
# How can I help you? your answers are not so good
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# Response=[{'generated_text': 'your answers are not so good. I am not sure if you are just lazy or what.'}] (-0.0504910945892334 seconds)
# How can I help you? that's not very nice
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# Response=[{'generated_text': "that's not very nice of you.\nI'm just being nice. I'm just trying to be a good friend.\nI"}] (-0.5467207431793213 seconds)
# How can I help you? how were you nice, in what way? justify yourself.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# Response=[{'generated_text': 'how were you nice, in what way? justify yourself. what did you do? how did you do it? what did you learn? what did you gain'}] (-0.5385606288909912 seconds)
# How can I help you? 
####################################
