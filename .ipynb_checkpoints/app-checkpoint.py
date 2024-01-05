import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

model=pk.load(open('model.pkl','rb'))
vectorizer=pk.load(open('vectorizer.pkl','rb'))
review=st.text_input('Enter a review')

if st.button('Predict'):
    
    review_vectorized = vectorizer.transform([review])
 
    result = model.predict(review_vectorized)
    
    st.write(result[0]) 

    if result[0]==0:
        st.write('This is a NEGATIVE review')
    else:
         st.write('This is a POSITIVE review')