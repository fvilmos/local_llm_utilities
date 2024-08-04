"""
Simple question and answer with local llm.

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

MODEL = cfg_data["LLM_MODEL"]

msg_sys="""You are a helpfull AI assistent that answers user question."""

if len(sys.argv)==1:
    print ("""\n***Provide a question as an argument!***\n
    Uses direcly the LLM model capabilities to answer a given question.
    usage: qa_simple.py \"Calculate 2+3*5\"\n\n""")

    exit()

else:
    question = sys.argv[1]
    answer = answer_a_question_msg([{'role': 'system', 'content': msg_sys},{"role": "user", "content": f'{question}'}],model=MODEL)

print ("\n***question***\n", sys.argv[1])
print ("\n***answer***\n",answer)

