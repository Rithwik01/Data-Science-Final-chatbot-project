import nltk #tokenizer library
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
import pickle 
import numpy as np

from keras.models import load_model
model = load_model('/Users/rithwikkamalesh/Downloads/chatbot-python-project-data-codes/chatbot_model.h5') #this loads the chatbot model
import json
import random
intents = json.loads(open('/Users/rithwikkamalesh/Downloads/chatbot-python-project-data-codes/intents.json').read())
words = pickle.load(open('/Users/rithwikkamalesh/Downloads/chatbot-python-project-data-codes/words.pkl','rb'))
classes = pickle.load(open('/Users/rithwikkamalesh/Downloads/chatbot-python-project-data-codes/classes.pkl','rb'))


def clean_up_sentence(sentence): #This whole method is essentially used to split every word and clean up the sentance

    # this is a tokenizer pattern, it is used to split words into an array
    sent_Words = nltk.word_tokenize(sentence)
    #this creates a shortform for each word
    sent_Words = [lemmatizer.lemmatize(word.lower()) for word in sent_Words]
    return sent_Words

# return group of words array: 0 or 1 for each word in the group that exists in the sentence

def bow(sentence, words, show_details=True):
    # breaking down the pattern
    sent_Words = clean_up_sentence(sentence)
    # creates a bag of words
    group = [0]*len(words)  
    for s in sent_Words:
        for i,w in enumerate(words):
            if w == s: 
                # this code is used to select words from the bag of words an
                group[i] = 1
                if show_details:
                    print ("found in group: %s" % w)
    return(np.array(group))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json): #Used to pick the correct response 
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents'] #words from the intent file
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses']) #uses the random fuction to pick up the correct response
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("red", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("Medicne chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="yellow", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("black",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
