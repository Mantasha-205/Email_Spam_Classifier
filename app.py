import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps= PorterStemmer()

import nltk
nltk.download('punkt')
nltk.download('stopwords')    #For streamlit ;requirements.txt

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)    #not the text is converted to list
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
           y.append(i) 
    text = y[:] 
    y.clear()
    for i in text:
        y.append(ps.stem(i)) 
    return " ".join(y)

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_email = st.text_area("Enter the message")

if st.button('Predict'):
    #1.Preprocess
    transformed_email= transform_text(input_email)
    #2.vectorize
    vector_input = cv.transform([transformed_email])
    #3.predict
    result = model.predict(vector_input)[0]
    #4.Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("ham")