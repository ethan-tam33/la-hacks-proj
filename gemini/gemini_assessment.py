"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

import google.generativeai as genai
from PIL import Image
import sys
import argparse #for command line arguments
import os
from dotenv import load_dotenv

load_dotenv("../.env")



parser = argparse.ArgumentParser(description='Process some command line inputs.')

# Adding arguments
#parser.add_argument('arg1', type=str, help='Llava descriptors')
parser.add_argument('arg1', type=str, help='Coral Image')
parser.add_argument('arg2', type=float, help='Health predictor score')
parser.add_argument('arg3', type=float, help='Confidence score')
args = parser.parse_args()

# img = Image.open("test_images/" + args.arg1)
img = Image.open(args.arg1)
health_score = args.arg2
confidence_score = args.arg3
#print(f"Llava descriptors: {args.arg1}")
#print(f"Health predictor score: {args.arg2}")

genai.configure(api_key=os.getenv("GEMINI_KEY"))

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

system_instruction = """A health score is a value 0 or 1, assigned to a given coral, where 0 represents that the coral is completely unhealthy 
and 1 represents completely healthy.
A confidence score is how probable or confident we are that the health score is correct (0 being most probable and 1 being least probable).
Your task is to analyze an image, asse the health score's correctness, and then once you confirm the health
score is correct, explain why it reached that health score. For example, if an image is assigned a 0 (unhealthy) and you see large amounts of 
color and fish in the coral, this image is likely wrong, and you should output that the classification has low confidence and may be incorrect. In the 
same vein, if the score is a 1 (healthy) and you see large amounts of white and bleaching, you should also output that there is low confidence in the 
classification and it might be incorrect. 
If you decide the classification is correct, you should  reason/explain why that health score might have been assigned based off features visible from the image. Keep responses to 1 paragraph of length. If there is no reason 
why the coral would be assigned the health score. Please only reason why it is correct if you believe the classfication is correct. Include scientific terminology. 
Make no mention of the numerical health score assigned to it, and never say the phrase health score. If you believe the model is incorrect, do not do this next step; if you believe it is correct, in your last sentence, take into consideration the confidence score. For example, if the health score is 1 and there is a confidence score of 0.50, and the image 
shows there is some white spots, perhaps the uncertaintiy is because of this. Only include analysis of uncertainty scoring in your last sentence. If you believe
the model is misclassifying, do not discuss the confidence score."""

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", #gemini-pro-vision
                              generation_config=generation_config,
                              system_instruction=system_instruction,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
])

convo.send_message(["The coral in the photo was assigned the following health score: "+ str(health_score) +", and the following confidence score: " + str(confidence_score) + ".", img])
sys.stdout.write(convo.last.text)




