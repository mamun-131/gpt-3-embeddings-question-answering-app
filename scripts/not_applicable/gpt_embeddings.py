# -*- coding: utf-8 -*-
"""
@author: Md Mamunur Rahman
"""

import openai
#import plotly.express as px
#from openai.embeddings_utils import get_embedding, cosine_similarity
import json
import pandas as pd
import numpy as np
import textract
import tiktoken

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"



def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))



def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.Context) for idx, r in df.iterrows()
    }

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

#f"Context separator contains {separator_len} tokens"

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len = len(document_section.Context.split(" "))
        #chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.Context.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
  #  print(f"Selected {len(chosen_sections)} document sections:")
  #  print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

#read openai api key from saved file
with open(r"gpt_key.txt", "r") as readfile:
    openai.api_key = json.load(readfile)["key"]

# read local msword file
context = str(textract.process('data/data1.docx')).replace("\\n\\n\\n", "^p^")#.replace("\\t","")
context = context.replace("\\n\\n",": ").replace("\\n","").replace("\\xe2\\x80\\x93","-").replace("\\t","")
#print(context)
result = list(filter(lambda x : x != '', context.split('^p^')))

df=pd.DataFrame()
df.insert(loc=0, column='Context', value=result)
#print(df)


#### save context embedding to json file
# document_embeddings = compute_doc_embeddings(df)
# #print(document_embeddings)
# df = pd.DataFrame(document_embeddings)
# #print (df)
# df.to_json('data/df_embeddings.json')
#########################################
df_emb=pd.DataFrame()
df_emb = pd.read_json('data/df_embeddings.json')



# lst=order_document_sections_by_query_similarity("when the file will be re-password protected?", df_emb)
# matching_context_indx = lst[0][1]
# print(lst[0][1])

# prompt_context = df.iloc[[matching_context_indx]]
# print(prompt_context)

# prompt = construct_prompt(
#     "when the file will be re-password protected?",
#     df_emb,
#     df
# )

#print("===\n", prompt)


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print("")

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")

print("Question: when the file will be re-password protected?")
print("Answer:  " + answer_query_with_context("when the file will be re-password protected?", df, df_emb))
