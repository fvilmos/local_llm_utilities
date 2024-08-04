"""
Simple question and answer based on an image input

Author: fvilmos
https://github.com/fvilmos
"""

import sys
import json
from utils.utils import *
from utils import tools

# load config data
jf = open(".\\data\\config.json",'r')
cfg_data=json.load(jf)

MODEL = cfg_data["VLLM_MODEL"]

msg_sys="""You are a helpfull AI assistent that answers user question."""

if len(sys.argv)==1:
    print ("""\n***Provide a question as an argument!***\n
    Uses a Vision-LLM to get the context about an image.
    usage: qa_with_vision.py "is a car on the image?" ".\data.test.jpg" \n
    or qa_with_vision.py "is a car on the image?" 0 \n""")

    exit()

elif len(sys.argv)==3:
    question = sys.argv[1]

    img_list = ['.\\data\\test.jpg']
    msg =[{'role':'user', "content":sys.argv[1], "images":[sys.argv[2]]} ]
    answer = answer_on_image_content(msg=msg, model='llava')
else:
    print ("Argument mismatch!")
    exit()

print ("\n***question***\n", sys.argv[1])
print ("\n***answer***\n",answer)

