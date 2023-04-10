# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:14:02 2023

@author: Md Mamunur Rahman
"""


import openai
import json
import pandas as pd
import numpy as np
import textract
import tiktoken
import os

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

basepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
with open(r"gpt_key.txt", "r") as readfile:
    apikye = json.load(readfile)["key"]
    openai.api_key  = apikye

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]



class GPTReadWriteData:
    def __init__(self,reading_path,writing_path):
        self.data_reading_master_path = reading_path
        self.data_writing_master_path = writing_path
    
    def data_reading(self,source_filename):
        # read local msword file
        context = str(textract.process(self.data_reading_master_path + '/' + source_filename)).replace("\\n\\n\\n", "^p^")#.replace("\\t","")
        #print(context)
        
        context = context.replace("\\n\\n",": ").replace("\\n","").replace("\\xe2\\x80\\x93","-").replace("\\t","").replace("\\xe2\\x80\\xa2"," - ")
        
        result = list(filter(lambda x : x != '', context.split('^p^')))
        return result
    
    def data_writing(self, source_filename):
        df = pd.DataFrame()
        result = self.data_reading(source_filename)
        df.insert(loc=0, column='Context', value=result)
        df.to_json(self.data_writing_master_path + '/' + 'df_paragraph_' + str(os.path.splitext(source_filename)[0]) + '.json')
        return 'df_paragraph_' + str(os.path.splitext(source_filename)[0]) + '.json'



class GPTEmbedding:
    def __init__(self,reading_path,writing_path):
        self.data_reading_master_path = reading_path
        self.data_writing_master_path = writing_path   
        
    def paragraph_reading_from_file(self,source_filename):
        df_par=pd.DataFrame()
        df_par = pd.read_json(self.data_reading_master_path + '/' + source_filename)
        return df_par
    
    def compute_doc_embeddings(self, df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
        
        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        return {
            idx: get_embedding(r.Context) for idx, r in df.iterrows()
        }
    def sava_embedded_data_into_file(self, filename, file_rewrite: bool):
        if file_rewrite == False or file_rewrite == None:
           if os.path.exists( self.data_writing_master_path + '/' + 'embedding_' + filename):
               print("file exists")
               return self.data_writing_master_path + '/' + 'embedding_' + filename
           else:
               document_embeddings = self.compute_doc_embeddings(self.paragraph_reading_from_file(filename))
               df = pd.DataFrame(document_embeddings)
               df.to_json(self.data_writing_master_path + '/' + 'embedding_' + filename)
               return self.data_writing_master_path + '/' + 'embedding_' + filename
        else:
            document_embeddings = self.compute_doc_embeddings(self.paragraph_reading_from_file(filename))
            df = pd.DataFrame(document_embeddings)
            df.to_json(self.data_writing_master_path + '/' + 'embedding_' + filename)
            return self.data_writing_master_path + '/' + 'embedding_' + filename     
        
class GPTQA:
     def __init__(self, paragraph_data_filename, embedding_filename,):
         self.MAX_SECTION_LEN = 500
         self.SEPARATOR = "\n* "
         self.ENCODING = "gpt2"  # encoding for text-davinci-003
         self.encoding = tiktoken.get_encoding(self.ENCODING)
         self.separator_len = len(self.encoding.encode(self.SEPARATOR))
         self.COMPLETIONS_API_PARAMS = {
             # We use temperature of 0.0 because it gives the most predictable, factual answer.
             "temperature": 0.0,
             "max_tokens": 300,
             "model": COMPLETIONS_MODEL,
         }
         self.df=pd.DataFrame()
         self.df = pd.read_json('data/' + paragraph_data_filename)
         self.df_emb=pd.DataFrame()
         self.df_emb = pd.read_json(embedding_filename)
         
     def vector_similarity(self,x: list[float], y: list[float]) -> float:
        """
        Returns the similarity between two vectors.
        
        Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
        """
        return np.dot(np.array(x), np.array(y))

     def order_document_sections_by_query_similarity(self,query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = get_embedding(query)
        
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)
        
        return document_similarities
             
     def construct_prompt(self, question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
            """
            Fetch relevant 
            """
            most_relevant_document_sections = self.order_document_sections_by_query_similarity(question, context_embeddings)
            
            chosen_sections = []
            chosen_sections_len = 0
            chosen_sections_indexes = []
             
            for _, section_index in most_relevant_document_sections:
                # Add contexts until we run out of space.        
                document_section = df.loc[section_index]
                
                chosen_sections_len = len(document_section.Context.split(" "))
                #chosen_sections_len += document_section.tokens + separator_len
                if chosen_sections_len > self.MAX_SECTION_LEN:
                    break
                    
                chosen_sections.append(self.SEPARATOR + document_section.Context.replace("\n", " "))
                chosen_sections_indexes.append(str(section_index))
                    
            # Useful diagnostic information
          #  print(f"Selected {len(chosen_sections)} document sections:")
          #  print("\n".join(chosen_sections_indexes))
            
            header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
            
            return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"         
         
     def answer_query_with_context(self,
            query: str,
            show_prompt: bool = False
        ) -> str:

                  
         prompt = self.construct_prompt(
                query,
                self.df_emb,
                self.df
            )
            
         if show_prompt:
                print("")
        
         response = openai.Completion.create(
                        prompt=prompt,
                        **self.COMPLETIONS_API_PARAMS
                    )
        
         return response["choices"][0]["text"].strip(" \n")        
