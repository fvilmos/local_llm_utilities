"""
Direct Retrival-Augmented Generation (RAG) - uses a source of input local(.txt, .pdf) or web (http://...) to enhance the input prompt of the LLM.
To create or update the local vector database use the "update_local_vector_db.py"

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

#print (tools.search_wikipedia("Population of france?"))
MODEL = cfg_data["LLM_MODEL"]

msg_sys="""You are a helpfull AI assistent that answers user question."""

if len(sys.argv)==1:
    print ("""\n***Provide a question as an argument!***\n
    usage:
        direct Retrival-Augmented Generation (RAG) - uses a source of input to enhance the input prompt of the LLM.
        Usage with local file: qa_direct_rag.py \"where was the G20 summit held in 2023?\" \".\data\wikiscrap.txt\"
        Usage with Wiki search: qa_direct_rag.py \"whre was the G20 summit held in 2023?\" \"https://en.wikipedia.org/wiki/G20\"
        \n\n""")
    exit()
elif len(sys.argv)==3:
    question = sys.argv[1]

    loader_type = 'text'

    # call over web
    if 'http' in str(sys.argv[2]):
        loader_type = 'web'

    answer = answer_with_rag(question, source=str(sys.argv[2]),model_name=MODEL, verbose=1, top_k=3, chunk_size=int(cfg_data['RAG_DEFAULT_CHUNK_SIZE']))
else:
    print ("Parameter number mismatch! Use a question and a source as input.")
    exit()

print ("\n***question***\n", sys.argv[1])
print ("\n***answer***\n",answer)

