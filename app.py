'''
#from chatbot import chatbot
from flask import Flask, render_template, request
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from flask import Flask, jsonify, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
#all packages 
import nltk 
import string 
import re
import random
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

app = Flask(__name__)
app.static_folder = 'static'

model = tf.keras.models.load_model('model/chatbot.h5')
tokenizer = joblib.load('model/tokenizer.joblib')
a_file = open("model/topic_answers.pkl", "rb")
dictionary = pickle.load(a_file)
le = joblib.load('model/newle.joblib')

stop_w = stopwords.words('english')
MAX_LEN = 200

def clean(column,df):
  df[column] = df[column].apply(str)
  df[column] = df[column].str.lower().str.split()
  #remove stop words
  df[column]=df[column].apply(lambda x: [item for item in x if item not in stop_w])
  #remove punctuation
  df[column]=df[column].apply(lambda x: [item for item in x if item not in string.punctuation])
  df[column]=df[column].apply(lambda x: " ".join(x))

def predictor(userText):
    data = [userText]
    x_try = pd.DataFrame(data,columns=['text'])
    clean('text',x_try)
    x_try = x_try['text']
    x_try = tokenizer.texts_to_sequences(x_try)
    x_try = pad_sequences(x_try, maxlen = MAX_LEN)
    ans = model.predict(x_try)
    result=[]
    for val in ans:
        res = val.argmax()
        result.append(res)
    prediction = le.inverse_transform(result)
    return prediction[0]

greetings = ["Hi", "Hi!","hey!", "Hello","Heyy","Hi.","Hey"]   
happy_emotions = ["I feel good","Life is good", "Life is great!","I've had a wonderful day!","I'm doing good"]
goodbyes = ["Bye!","Thanks and bye","Ok thanks! Bye!","Goodbye","See ya later","Alright thanks. Bye","That's all. Bye!"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if userText in greetings:
        return "Hello! How may I help you today?"
    elif userText in happy_emotions:
        return "That's great! Do you still have any questions for me?"
    elif userText in goodbyes:
        return "Hope I was able to help you today! Take care, bye!"
    topic = predictor(userText)
    res = random.choice(dictionary[topic])
    return res
    

if __name__ == "__main__":
    app.run() 
'''
from bert_embedding import BertEmbedding
#from bert_serving.client import BertClient
from flask import Flask, render_template, request
import os
import json
import requests
import pickle
import joblib
import numpy as np
import pandas as pd
#import tensorflow as tf
#all packages 
import nltk 
import string 
import re
import random
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from textblob.sentiments import *
import re
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

app = Flask(__name__)
app.static_folder = 'static'

#a_file = open("model/topic_answers.pkl", "rb")
#dictionary = pickle.load(a_file)
sent_bertphrase_embeddings = joblib.load('model/questionembedding.dump')
sent_bertphrase_ans_embeddings = joblib.load('model/ansembedding.dump')

stop_w = stopwords.words('english')
bert_embedding = BertEmbedding(max_seq_length=50)
#bc = BertClient(ip='localhost')
df = pd.read_csv("model/20200325_counsel_chat.csv",encoding="utf-8")

def get_embeddings(texts):
    url = '5110e4cb8b63.ngrok.io'
    headers = {
        'content-type':'application/json'
    }
    data = {
        "id":123,
        "texts":texts,
        "is_tokenized": False
    }
    data = json.dumps(data)
    r = requests.post("http://"+url+"/encode", data=data, headers=headers).json()
    return r['result']

def clean(column,df,stopwords=False):
  df[column] = df[column].apply(str)
  df[column] = df[column].str.lower().str.split()
  #remove stop words
  if stopwords:
    df[column]=df[column].apply(lambda x: [item for item in x if item not in stop_w])
  #remove punctuation
  df[column]=df[column].apply(lambda x: [item for item in x if item not in string.punctuation])
  df[column]=df[column].apply(lambda x: " ".join(x))

def retrieveAndPrintFAQAnswer(question_embedding,sentence_embeddings,FAQdf):
    max_sim=-1
    index_sim=-1
    valid_ans = []
    for index,faq_embedding in enumerate(sentence_embeddings):
        #sim=cosine_similarity(embedding.reshape(1, -1),question_embedding.reshape(1, -1))[0][0];
        sim=cosine_similarity(faq_embedding,question_embedding)[0][0]
        #print(index, sim, sentences[index])
        if sim>=max_sim:
            max_sim=sim
            index_sim=index
            valid_ans.append(index_sim) #get all possible valid answers with same confidence

    #Calculate answer with highest cosine similarity 
    max_a_sim=-1
    answer=""
    for ans in valid_ans:
      answer_text = FAQdf.iloc[ans,8] #answer 
      answer_em = sent_bertphrase_ans_embeddings[ans] #get embedding from index
      similarity = cosine_similarity(answer_em,question_embedding)[0][0]
      if similarity>max_a_sim:
        max_a_sim = similarity
        answer = answer_text
    #print("\n")
    #print("Question: ",question)
    #print("\n");
    #print("Retrieved: "+str(max_sim)+" ",FAQdf.iloc[index_sim,3])  # 3 is index for q text
    #print(FAQdf.iloc[index_sim,8])    # 8 is the index for the answer text 
    #check confidence level
    if max_a_sim<0.70: 
        return "Could you please elaborate your situation more? I don't really understand." 
    return answer  
    #print(answer)

def retrieve(sent_bertphrase_embeddings,example_query):
    max_=-1
    max_i = -1
    for index,emb in enumerate(sent_bertphrase_embeddings):
        sim_score = cosine_similarity(emb,example_query)[0][0]
        if sim_score>max_:
            max_=sim_score
            max_i=index
    #print("\n");
    #print("Retrieved: "+str(max_)+" ",df.iloc[max_i,3])  # 3 is index for q text
    #print(df.iloc[max_i,8])    # 8 is the index for the answer text  
    return str(df.iloc[max_i,8])

def clean_text(greetings):
    greetings = greetings.lower()
    greetings = ' '.join(word.strip(string.punctuation) for word in greetings.split())
    re.sub(r'\W+', '',greetings)
    greetings = lmtzr.lemmatize(greetings)
    return greetings

def predictor(userText):
    data = [userText]
    x_try = pd.DataFrame(data,columns=['text'])
    #clean the user query
    clean('text',x_try,stopwords=True)
    #x_try = x_try['text']
    #get BERT embedding of question
    '''
    x_try_list = x_try['text'].to_list()
    #x_try_embeddings = bert_embedding(x_try_list)
    user_queries = []
    for x in x_try_embeddings:
        user_queries.append(x[1])
    for queries in range(len(user_queries)):
        #print(x_try.iloc[queries][0])
        example_query = user_queries[queries]
        return retrieve(sent_bertphrase_embeddings,example_query)
    '''
    for index,row in x_try.iterrows():
        question = row['text']
        #question_embedding = bert_embedding(question)
        #question_embedding = question_embedding[0][1]
        question_embedding = get_embeddings([question])
        #question_embedding = bc.encode([question])
        return retrieveAndPrintFAQAnswer(question_embedding,sent_bertphrase_embeddings,df)
    '''
    ans = model.predict(x_try)
    result=[]
    for val in ans:
        res = val.argmax()
        result.append(res)
    prediction = le.inverse_transform(result)
    return prediction[0]
    '''
greetings = ['hi','hey', 'hello', 'heyy', 'hi', 'hey', 'good evening', 'good morning', 'good afternoon', 'good', 'fine', 'okay', 'great', 'could be better', 'not so great', 'very well thanks', 'fine and you', "i'm doing well", 'pleasure to meet you', 'hi whatsup']
happy_emotions = ['i feel good', 'life is good', 'life is great', "i've had a wonderful day", "i'm doing good"]
goodbyes = ['thank you', 'thank you', 'yes bye', 'bye', 'thanks and bye', 'ok thanks bye', 'goodbye', 'see ya later', 'alright thanks bye', "that's all bye", 'nice talking with you', 'i’ve gotta go', 'i’m off', 'good night', 'see ya', 'see ya later', 'catch ya later', 'adios', 'talk to you later', 'bye bye', 'all right then', 'thanks', 'thank you', 'thx', 'thx bye', 'thnks', 'thank u for ur help', 'many thanks', 'you saved my day', 'thanks a bunch', "i can't thank you enough", "you're great", 'thanks a ton', 'grateful for your help', 'i owe you one', 'thanks a million', 'really appreciate your help', 'no', 'no goodbye']
'''
greetings = ["hi", "hi!","hey!", "hello","heyy","hi","hey","good evening","good morning","good afternoon",
"good","fine","okay","great","could be better","not so great","very well, thanks","fine, and you?","I'm doing well",
"pleasure to meet you","hi whatsup"]   
happy_emotions = ["I feel good","Life is good", "Life is great!","I've had a wonderful day!","I'm doing good"]
goodbyes = ["thank you","Thank you","Yes bye","Bye!","Thanks and bye","Ok thanks! Bye!","Goodbye","See ya later","Alright thanks. Bye","That's all. Bye!",
"Nice talking with you","I’ve gotta go","I’m off","good night!","see ya!","see ya later!","Catch ya later","Adios","Talk to you later",
"Bye bye!","All right then","thanks","Thank you!","Thx","Thx bye","thnks","thank u for ur help",
"many thanks","you saved my day","Thanks a bunch","I can't thank you enough","you're great",
"thanks a ton","grateful for your help","i owe you one","thanks a million","really appreciate your help","No","No, goodbye"]
'''
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    cleanText = clean_text(str(userText))
    #check sentiment 
    blob = TextBlob(userText, analyzer=PatternAnalyzer())
    polarity = blob.sentiment.polarity

    if cleanText in greetings:
        return "Hello! How may I help you today?"
    elif polarity>0.7:
        return "That's great! Do you still have any questions for me?"
    elif cleanText in happy_emotions:
        return "That's great! Do you still have any questions for me?"  
    elif cleanText in goodbyes:
        return "Hope I was able to help you today! Take care, bye!"
    topic = predictor(userText)
    #res = random.choice(dictionary[topic])
    return topic

if __name__ == "__main__":
    app.run() 
