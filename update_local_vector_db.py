"""
Update or create a local vector database, location is specified in the config.json file, default .vstorage

Input is a list of sorces, can be pdf, txt, or web sources
i.e. \\intputs\\G20.pdf" "https://www.wikipedia.org/g20

Author: fvilmos, https://github.com/fvilmos
"""
import json
import sys
from utils.local_vector_db import LocalVectorDB

# load config data
jf = open(".\\data\\config.json",'r')
cfg_data=json.load(jf)


sources = ['.\\inputs\\Object Detection in 20 Years A Survey.pdf', '.\\inputs\\G20 - Wikipedia.pdf', '.\\inputs\\GLO-SD-EP-0028_Software Coding Standard.pdf','.\\inputs\\How to Verify SW Modules and Units.pdf']

if len(sys.argv)==1:
    print ("""usage: RAG with local db (persistent), update_local_vector_db.py \\data\\wikiscrap.txt https://www.wikipedia.org/g20 ... \n""")
    exit()
elif len(sys.argv)>1:
    
    # format input as a list
    in_len = len(sys.argv)
    in_list = sys.argv[1:in_len]

    if isinstance(in_list,list):

        # create local vector db
        db = LocalVectorDB(cfg_data["LOCAL_VECTOR_DB_PATH"], cfg_data["LOCAL_VECTOR_DB_NAME"])
        db.add_to_vector_db(in_list, chunk_size=cfg_data["RAG_DEFAULT_CHUNK_SIZE"],chunk_overlap=cfg_data["RAG_DEFAULT_OVERLAP_SIZE"])
    else:
        print ("wrong input type! Must be a list of strings.")

print ("Done!")




