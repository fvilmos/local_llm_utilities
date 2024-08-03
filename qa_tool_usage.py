"""
Question and answer with function call for tool usage.
Tools are automatically collected from the utils.tools file. 
An ollama tool signature is created from the prototypes and parameters, then is passed to the LLM.

Author: fvilmos
https://github.com/fvilmos
"""
import sys
import json
from utils.utils import *
from utils import tools
from utils.prompts import PROMPT_TEMPLATE

# load config data
jf = open(".\\data\\config.json",'r')
cfg_data=json.load(jf)

MODEL=cfg_data['LLM_MODEL']
MAX_TRIES=5
VERBOSE=0

# process available tools
tools_names = get_tools()
tools_functions = [eval('tools.'+ str(n)) for n in tools_names]
tools_json_signatures = [json.loads(function2json(f)) for f in tools_functions]
if VERBOSE:
    print ("available tools:",tools_names)

system_msg = f"""You are a helpful AI assitant, answer the user questions as best you can. DO NOT generate code in the answer. 
Always check the wikipedia first if the question is complex.
\n Available tools: {tools_names}\n

"""

if len(sys.argv)==1:
    print ("""\n***Provide a question as an argument!***\n
    usage:
        LLM use warious tools to answer the question: qa_tool_usage.py \"where was the G20 summit held in 2023?\" 
        \n""")
    exit()

elif len(sys.argv)==2:
    question = sys.argv[1]

    init_prompt = [{"role": "system", "content": system_msg},{"role":"user", "content":question + "\nLet's think step by step."}]
    while MAX_TRIES>0:

        answer = answer_a_question_msg(msg=init_prompt,model=MODEL, tool_list=tools_json_signatures, return_raw=True)

        if 'tool_calls' in answer['message']:
            for t in answer['message']['tool_calls']:
                
                # get function elements
                f_name = t['function']['name']
                f_params = t['function']['arguments']

                # validate param types
                fn_signature = json.loads(function2json(eval('tools.' + f_name)))
                fn_val_prop = fn_signature['function']['parameters']['properties']
                
                # create argument list
                arg_list = [f'{a[0]}' for a in dict(f_params).items()]

                arg_type_list = []
                for a in arg_list:
                    if a in fn_val_prop:
                        arg_type = fn_val_prop[str(a)]['type']
                        arg_type_list.append(arg_type)
                    else:
                        pass

                # generate arg string
                arg_string = ""
                for i,a in enumerate(dict(f_params).items()):
                    if arg_type_list[i]=='str':
                        arg_string += f'{a[0]}="{a[1]}", '
                    else:
                        arg_string += f'{a[0]}={a[1]}, '

                # function call
                f_return = ''
                try:
                    f_return = eval(f'tools.{f_name}({arg_string})')
                    if VERBOSE:
                        print (f'function call: tools.{f_name}({arg_string}) => Return:{f_return}\n')

                except BaseException as err:
                    print(f"Unexpected {err.args}")
                
                # add answer to the prompt and call the LLM with the enhanced prompt
                answer_chunk = f"\nResult of the tool usage: {f_return}\n"
                init_prompt.append({"role": "assistant", "content": answer_chunk})
        else:
            break
        MAX_TRIES -=1
        
        if VERBOSE:
            print ("========\nIteration\n========\n:", MAX_TRIES)

print ("\n***question***\n", sys.argv[1])
print ("\n***answer***\n",answer["message"]["content"])
