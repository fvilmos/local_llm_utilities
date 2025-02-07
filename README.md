### Local Large Language Model utilities

This set of utilities is focused on Local LLM usage with ollama server [1]. The main goal is to decouple the implementations from online access or pre-paid services as much as possible.

List of features:
- create / update local vector database(es) using Chroma [4]
- Local  Retrieval-Augmented Generation (RAG) with Chroma for (.txt,.pdf) or webpage
- Direct RAG - by digesting directly different filetypes (.txt,.pdf) or webpage ** takes time
- LLM tool usage (function calls), and an easy base to extend with specific implementations
- usage of Vision-LLMs (image to text description) to describe a scene, input can be a stored image or from source i.e. webcam
Currently available tools (utils.tools):  get_local_datetime, calculate, search_wikipedia

#### Prerequisites

1. install ollama[1]
2. pull ollama moldes: llama3.1 [2], nomic-embed-text[5]
```NOTE: you can use different LLMs from the ollama model zoo, but, for tool usage / function call is needed one that support this feature i.e. llama3.1, see 'qa_tool_usage.py' example```
3. install requirements ```pip -r requirements.txt```

#### Example usage

1. Image to text - scene description

<p align="center"> 
  <img src="data/test.jpg" alt="" width="320"></a>
  <div align="center">input image</div>
</p>

```shell
qa_with_vision.py "Describe the image and provide a list of detected objects with their bounding box coordinates like: x,y,w,h" ".\data\test.jpg"

***question***
 Describe the image and provide a list of detexted objects and their bounding box coordinates like: x,y,w,h

***answer***
  The image shows a collection of items on a surface, with a smartphone and remote control in the background. In the foreground, there are two small containers that appear to be tubes or capsules, one 
is orange and the other is blue. These containers are placed next to each other.
Here's a list of detected objects and their bounding box coordinates:
1. Orange container: [0.356,0.497,0.528,0.761]
2. Blue container: [0.500,0.497,0.634,0.761]
3. Remote control: [0.000,0.000,0.338,0.909]
4. Smartphone: [0.328,0.059,0.998,0.986]
Please note that the bounding box coordinates are approximate and may not be entirely accurate due to the resolution of the image and the complexity of the objects' shapes.
```

```
NOTE: to use the webcam call the scipt with the devide number. i.e.
qa_with_vision.py "Describe the image and provide a list of detected objects 
with their bounding box coordinates like: x,y,w,h" 0 
```


2. Simple question and answer with local llm.
```shell
qa_simple.py "where was the G20 summit held in 2023? Provide the country and localtion."    

***question***
 where was the G20 summit held in 2023? Provide the country and localtion.
***answer***
 I'm not aware of any information about a specific G20 summit being held in 2023, as my training data only goes up to 2022. However, I can suggest some possible sources where you might be able to find this information:

1. Official G20 website: You can check the official G20 website (g20.org) for updates on upcoming summits.
2. News websites and online media outlets: Websites like BBC, CNN, Al Jazeera, and other reputable news sources often provide coverage of international events like the G20 summit.

If you have any more information or context about the 2023 G20 summit, I'd be happy to try and help you find the answer!
===Not correct===
```
3. Question and answer with function call for tool usage.

```shell
qa_tool_usage.py "where was the G20 summit held in 2023? Provide the country and localtion."

***question***
 where was the G20 summit held in 2023? Provide the country and localtion.
***answer***
 meetings, and working groups.
The 2023 G20 summit was held in New Delhi, India.
===Correct===
```
4. Digest pdf file and save vectors in chroma locally, than ask based on the new knowledge

```shell
update_local_vector_db.py .\\data\\catan_base_rules_2020_200707.pdf
input type: pdf,  .\\data\\catan_base_rules_2020_200707.pdf
Done!

qa_local_db_rag.py "what resource cards are needed to build a settlement in CATAN? Provide the type and the number."

***question***
 what resource cards are needed to build a settlement in CATAN? Provide the type and the number.
***answer***
 According to the text, to build a Settlement in Catan, you need 1 Brick, 1 Wood (or Lumber), 1 Sheep (or Wool), and 1 Wheat (or Grain).
```

#### Resources
1. [Ollama - local model server](https://ollama.com/)
2. [llama3.1 new state-of-the-art model from Meta](https://ollama.com/library/llama3.1)
3. [Langchain documentation](https://python.langchain.com/v0.2/docs/introduction/)
4. [Chroma](https://www.trychroma.com/)
5. [A high-performing open embedding model with a large token context window](https://ollama.com/library/nomic-embed-text)

/Enjoy.
