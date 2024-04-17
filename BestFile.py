#Final Project

#Importing all the required libraries
from fastapi import FastAPI
import spacy  # It is a open source python library. It is a pretrained model.
from sklearn.metrics.pairwise import cosine_similarity # It is used to calculate cosine similarity bewtween two vectors which we are going to get after passing through our model
import pandas as pd   #Just to read our csv file
import en_core_web_sm
# from mangum import Mangum

app = FastAPI()
# handler = Mangum(app)

nlp_model = en_core_web_sm.load() #The model which we are using

def preprocessing(text):
        doc = nlp_model(text)   ##  Here we do tokenization
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  ##Here we iterate through each token and check for stop word and punctuation
        
        return " ".join(tokens) # Here we return all the tokenized elements of strings seperated by comma
        
def calculate_similarity(text1, text2):  # Here we are taking two text as the input parameter of whom we are going to calculate similarity.

    pre_process1 = preprocessing(text1)  ## Here we are tokenizing the input text by calling above function
    pre_process2 = preprocessing(text2)


    document1 = nlp_model(pre_process1) ## Here we are doing vector representation of tokens
    document2 = nlp_model(pre_process2)
    embedded1 = [token.vector for token in document1] # here we get the list of word emeddings for text 1
    embedded2 = [token.vector for token in document2] # here we get the list of word emeddings for text 2

    # Calculate sentence embeddings (average of word embeddings)
    text_embedded1 = sum(embedded1) / len(embedded1) ## By averaging the word emeddings we calculate text embeddings for text 1
    text_embedded2 = sum(embedded2) / len(embedded2) ## By averaging the word emeddings we calculate text embeddings for text 2

    # Calculate cosine similarity
    similarityscore = cosine_similarity([text_embedded1], [text_embedded2])[0][0]
    return similarityscore



@app.post("/similarity/")  
async def similarity(text1: str, text2: str):  #Taking_input
    Score = calculate_similarity(text1, text2)
    return {"Similarity_Score": str(Score)}