
#Load packages
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

#Package for regular expressions
import re

st.title("Fake News Detector")

#Loading training and testing dataset
df = pd.read_csv("COVID_News.csv")
df['Content'] = df['Content'].astype(str)

#Defining X & Y
X = df["Content"]
Y = df["Label"].tolist()

# In[18]:
#Cleaning content and splitting into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)


# In[19]:


#Vectorizing content using TFIDF
tf_idfvectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
X_train_vectorized = tf_idfvectorizer.fit_transform(X_train)


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(256,8,8), activation = 'relu', solver='adam',random_state=1)
clf.fit(X=X_train_vectorized,y=Y_train)
print("Training accuracy",clf.score(X=X_train_vectorized,y=Y_train))

strng = st.text_input("Enter Text :")
strng=[strng]


X_test_vectorized = tf_idfvectorizer.transform(strng)

#predicting the sentiments for the test dataset
Y_pred = clf.predict(X_test_vectorized)

if Y_pred == 0:
    st.subheader("✅ This seems real!👍")
else:
    st.subheader("❌ This seems fake!😡")


st.markdown("<br><br><hr><center><span style='color:red'>App created by PGPX Group 2 as a requirement for DSB course offered by professor Ankur Sinha</strong></a></center><hr>", unsafe_allow_html=True)