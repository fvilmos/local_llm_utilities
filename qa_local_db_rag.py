"""
RAG with with local vector database. 
To create or update the local vector database use before the "update_local_vector_db.py"

Author: fvilmos
https://github.com/fvilmos
"""

import sys
import json
from utils.utils import *


# load config data
jf = open(".\\data\\config.json",'r')
cfg_data=json.load(jf)

MODEL = cfg_data["LLM_MODEL"]

msg_sys="""You are a helpfull AI assistent that answers user questions."""

if len(sys.argv)==1:
    print ("""\n***Provide a question as an argument!***\n
    usage: RAG with local db (persistent), qa_local_db_rag.py \"where was the G20 summit held in 2023?\"\n\n""")
    exit()
elif len(sys.argv)==2:
    question = sys.argv[1]
    answer = answer_with_rag_db(question, db_path= cfg_data["LOCAL_VECTOR_DB_PATH"],model_name=MODEL, verbose=0, top_k=3)

print ("\n***question***\n", sys.argv[1])
print ("\n***answer***\n",answer)

