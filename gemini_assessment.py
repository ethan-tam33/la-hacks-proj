"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

import google.generativeai as genai

from PIL import Image
import sys
import argparse #for command line arguments

parser = argparse.ArgumentParser(description='Process some command line inputs.')

# Adding arguments
#parser.add_argument('arg1', type=str, help='Llava descriptors')
parser.add_argument('arg1', type=str, help='Coral Image')
parser.add_argument('arg2', type=float, help='Health predictor score')
args = parser.parse_args()

img = Image.open("test_images/" + args.arg1)
health_score = args.arg2
#print(f"Llava descriptors: {args.arg1}")
#print(f"Health predictor score: {args.arg2}")

genai.configure(api_key="AIzaSyBLM0Bv2qkWcE9JnGFflxMVvN8Q6_h4Eq8")

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
and 1 represents completely healthy. Your task is to analyze an images and assess it's correctness. Given such a score, and an image correspending to
Given such a score, and an image corresponding 
to the coral assigned said score, reason/explain why that health score might have been assigned based off features visible from the image. Keep responses to 1 paragraph of length. If there is no reason 
why the coral would be assigned the health score, please only state that there is a potential mischracterization. For example, if the reef is 
vibrant colors and assigned 0, there is likely a mischaracterization; if the reef is white with no life and assigned a 1, there is also likely a 
mischaracterization. Only reason based on described features that are relevant for coral health. Include scientific terminology. Make no mention of
 the numerical health score assigned to it, and never say the phrase health score."""

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", #gemini-pro-vision
                              generation_config=generation_config,
                              system_instruction=system_instruction,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
])

convo.send_message(["The coral in the photo was assigned the following health score: "+ str(health_score) +".", img])
sys.stdout.write(convo.last.text)