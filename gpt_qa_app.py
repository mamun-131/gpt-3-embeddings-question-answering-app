# -*- coding: utf-8 -*-

from scripts.gpt_classes import GPTReadWriteData, GPTEmbedding, GPTQA


####### Data reading and Writing ############
#data reading and writing class instantiating 
gptrw = GPTReadWriteData("data","data")
#writing daata into json file reading from source file in param
datafile = gptrw.data_writing("mamun.docx")
print (datafile)
#############################################

####### Data Embedding and save to file #####
gptemb = GPTEmbedding("data","data")
embedding_file = gptemb.sava_embedded_data_into_file(datafile, False)
print(embedding_file)

#############################################

####### Question and Answer #################
gptqa = GPTQA(datafile, embedding_file)

while True:
    user_input = input("You: ")
    if user_input == "bye":
        break
    prompt = "Chatbot: "
    response = prompt + gptqa.answer_query_with_context(user_input)
    print(response)
    print("")

#############################################