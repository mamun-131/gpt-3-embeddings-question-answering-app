# GPT-3/3.5/4 document Embeddings, Semantic Search, Question Answering Application

## Running Example of Python and OpenAI API

This is an app for document reading and asking question about the document. This app is powered by LLM (GPT-3.5). There is no direct question answering solution by GPT
for a bigger document. GPT context has limitation of 2049 tokens. So we have to apply a tric to split document into different paragraph and embadde the
paragraphs. When any question is ask that would be converted to word embedding and perform a semantic search on embeddings of full document and will find most relevent 
paragraph. The most relevent paragraph will be injected into chatgpt prompt through API and will get a generative answer. This process has an awesome result.

## Table of Contents
- [GPT-3.5 Question Answer App](#GPT-3.5 Question Answer App)
    - [Table of Contents](#table-of-contents)
    - [Overview](#overview)
    - [Development](#development)
    - [Install Dependencies](#install-dependencies)
	-[Outcomes](#outcomes)

## Overview
This app has used openai Embedding and Prompt engineering. I have taken the base code taken from the OpenAI API coding examples. I have appliced 
object orinted approach to make all codes more structured and better usable. I have used my resume as an example of data in data folder. Datafram paragraph and 
embedding files are also saved in the same folder. If you use this code you have to apply your own access key in gpt_key.text file as json format like 
{"key" : "sk-cQDfQCzd5Hxxxxxx4YkxCT3BlbkFxxxxxxxxxxxxxxxxx"}.  

## Development
There are 2 major files gpt_qa_aoo.py and gpt_class.py in script folder. You to run gpt_qa_aoo.py to runn the app. You can apply this files to your 
own chatbot framework (RASA, Dialogflow, MS Chatbot, etc)

## Install Dependencies
The following packages are needed to be installed by PIP :

- openai
- json
- pandas as pd
- numpy as np
- textract
- tiktoken

## Outcomes 
You can ask question about the resume like..
1. what is his email address?
2. what is his phone number?
3. what the are main skills?
4. what are the experiences in 2020?
5. what are his experiences in ML

You will get a surprise. :)
![Alt text](output.PNG?raw=true "Outcomes")
 
