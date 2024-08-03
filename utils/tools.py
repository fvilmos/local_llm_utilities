"""
Utility file, that implements tools for a ReAct agent
Function names for the file will be used to create dynamically the available tools list.
Function comments will be used to create the prompt for tool usage

NOTE: All imports from this module scope shall be hidden (under function) othervise will be imported in the global scope!

Author: fvilmos
https://github.com/fvilmos
"""

def get_local_datetime()->str:
    """
    Provide hte local date and time.
    example: get_local_datetime()
    returns "2022 Jun 20, 17:30:20".   
    """
    import time
    return str(time.ctime())

def calculate(expr:str)->str:
    """
    evaluats matematical expressions and caluclates the result.
    example: calculate ("5 * 7 + 3")
    returns 38
    """
    return eval(expr)

def search_wikipedia(search:str)->str:
    """
    Searched the wikipedia for the search input:
    example:search_wikipedia("the capital of France") 
    returns Paris
    """

    #modified from : https://api.wikimedia.org/wiki/Searching_for_Wikipedia_articles_using_Python
    import requests
    import json
    language_code = 'en'
    number_of_results = 1


    base_url = 'https://api.wikimedia.org/core/v1/wikipedia/'
    endpoint = '/search/page'
    url = base_url + language_code + endpoint
    parameters = {'q': search, 'limit': number_of_results}
    response = requests.get(url, params=parameters)

    # process results
    response = json.loads(response.text)
    return str(response['pages'][0]['excerpt'].strip().replace('<span class="searchmatch">','').replace('</span>','').replace('(&quot;','').replace('.&quot;)',''))





 

