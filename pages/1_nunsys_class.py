#libraries and bibliotecas
import pandas as pd
import numpy as np
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from haversine import haversine
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import plotly.express as px
import folium
from streamlit_folium import folium_static
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

#st.set_page_config (page_title = 'nunsys class', layout = 'wide')
############################################################################## 
# 1. Leemos los ficheros csv que necesitamos para realizar la tarea
##############################################################################
#path = 'D:/CAEM/NLP/'
train = 'train.csv'
test = 'test.csv'
sample = 'sample_submission.csv'
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
submission_data = pd.read_csv('sample_submission.csv')
# Archivos para dataframe... cal
test_010_cal = pd.read_csv ('test_010_cal.csv')
test_015_cal = pd.read_csv ('test_015_cal.csv')
test_020_cal = pd.read_csv ('test_020_cal.csv')
test_025_cal = pd.read_csv ('test_025_cal.csv')
test_030_cal = pd.read_csv ('test_030_cal.csv')
# Archivos para dataframe... val
test_010_val = pd.read_csv ('test_010_val.csv')
test_015_val = pd.read_csv ('test_015_val.csv')
test_020_val = pd.read_csv ('test_020_val.csv')
test_025_val = pd.read_csv ('test_025_val.csv')
test_030_val = pd.read_csv ('test_030_val.csv')
############################################################################## 
# 2. Realizamos un análisis previo de los datos, tal y cual se obtuvieron
##############################################################################
#Todo lo relativo a entrenamiento
twits_train_lon = len (train_data['text'])

print (twits_train_lon)
train_data['palabras'] = train_data['text'].str.split().str.len()
mean_words = train_data['palabras'].mean()
min_words = train_data['palabras'].min()
max_words = train_data['palabras'].max()
sd_words = train_data['palabras'].std()


twits_train_1 = train_data.loc[train_data['target']==1]
mean_words_1 = twits_train_1['palabras'].mean()
min_words_1 = twits_train_1['palabras'].min()
max_words_1 = twits_train_1['palabras'].max()
sd_words_1 = twits_train_1['palabras'].std()

twits_train_0 = train_data.loc[train_data['target']==0]
mean_words_0 = twits_train_0['palabras'].mean()
min_words_0 = twits_train_0['palabras'].min()
max_words_0 = twits_train_0['palabras'].max()
sd_words_0 = twits_train_0['palabras'].std()


#relativo al dataframe de test (validación)
twits_test = len (test_data['text'])
print (twits_test)

test_data['palabras'] = test_data['text'].str.split().str.len()
mean_words_v = test_data['palabras'].mean()
min_words_v = test_data['palabras'].min()
max_words_v = test_data['palabras'].max()
sd_words_v = test_data['palabras'].std()

# cáclulo de porcentajes relativos
porc1 = round(((len(train_data.loc[train_data['target']==1])/(len (train_data['text'])))*100),3)

porc0 = round(((len(train_data.loc[train_data['target']==0])/(len (train_data['text'])))*100),3)

porc90 = int (0.9*len (train_data['text']))
porc85 = int (0.85*len (train_data['text']))
porc80 = int (0.8*len (train_data['text']))
porc75 = int (0.75*len (train_data['text']))
porc70 = int (0.7*len (train_data['text']))

porc10 = int (0.1*len (train_data['text']))
porc15 =int ( 0.15*len (train_data['text']))
porc20 = int (0.20*len (train_data['text']))
porc25 = int (0.25*len (train_data['text']))
porc30 = int (0.3*len (train_data['text']))
#-------------------------------------------
#Barra Lateral Streamlit
#-------------------------------------------

image = Image.open("CE.jpg")
st.sidebar.image (image, width=120)

st.markdown("<h1 style='text-align: justify;'>Datos crudos de entrenamiento para la clasificación de tweets relativos a desatres naturales</h1>", unsafe_allow_html=True)
#st.header('Datos crudos para entrenamiento en la clasificación de tweets: desatres naturales')
st.sidebar.markdown ("# NUNSYS")
st.sidebar.markdown ("## Prueba Técnica para Senior DS")
st.sidebar.markdown ("""---""")
st.sidebar.markdown ('<p style="text-align: justify;">Esta página muestra como se entrena y valida varios modelos para discriminar tweets, clasificándolos en relativos o no a desastres naturales.</p>', unsafe_allow_html=True)
st.sidebar.markdown ('<p style="text-align: justify;">En la primera sección se muestra un análisis previo de los datos en crudo. Se visualiza además, la cantidad de datos en los ficheros de entrenamiento y de validación, la cantidad de datos que se utilizan para las distintas proporciones de partición seleccionadas, y la estadística básica del número de palabras en los tweets.</p>', unsafe_allow_html=True)
st.sidebar.markdown ('<p style="text-align: justify;">Las porporciones seleccionadas, para entrenamiento-testeo han sido de 90-10, 85-15, 80-20, 75-25, 70-30. Los modelos seleccionadas han sido el de Naive Bayes, Regresión Logística, SVM y Random Forest. Los índices de bondad son: Accuracy, Precision, Recall y F1-Score </p>', unsafe_allow_html=True)
st.sidebar.markdown ("""---""")
st.sidebar.markdown ('<p style="text-align: justify;">En la segunda sección se muestran los modelos utilizados y los resultados obtenidos con el entrenamiento y la validación del modelo.</p>', unsafe_allow_html=True)
st.sidebar.markdown ('<p style="text-align: justify;">Los cuatro algoritmos seleccionados han sido entrenados y testados en el dataset train. Se ha evaluado el desempeño de los mismos mediante los índices de bondad para cada algoritmo, además de distintas proporciones para entrenamiento-testeo ya definidas previamente</p>', unsafe_allow_html=True)
st.sidebar.markdown ("""---""")
st.sidebar.markdown ('<p style="text-align: justify;">Finalmente, en la tercera sección se muestra la validación del modelo en un fichero sin clasificación previa, llamado test.csv.</p>', unsafe_allow_html=True)

st.dataframe(train_data)
#-------------------------------------------
#Layout Streamlit
#-------------------------------------------
tab1,tab2,tab3 = st.tabs(['Datos originales', 'Resultados de entrenamiento', 'Validación en test.csv'])

with tab1:
    with st.container():
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.markdown("##### Número de datos (tweets) en fichero de entrenamiento")
            st.markdown(f"<h1 style='text-align: center;'>{twits_train_lon}</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown("##### Número de datos (tweets) en fichero de validación")
            st.markdown(f"<h1 style='text-align: center;'>{twits_test}</h1>", unsafe_allow_html=True)
        with col3:
            st.markdown("##### Tweets de alerta positiva de entrenamiento en %")
            st.markdown(f"<h1 style='text-align: center;'>{porc1}</h1>", unsafe_allow_html=True)
        with col4:
            st.markdown("##### Tweets de alerta negativa de entrenamiento en %")
            st.markdown(f"<h1 style='text-align: center;'>{porc0}</h1>", unsafe_allow_html=True)      
        st.markdown ("""---""") 
    with st.container():
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.markdown("##### Número de datos entrenamiento-validación para 90-10")
            st.markdown(f"<h1 style='text-align: center;'>{porc90} {porc10}</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown("##### Número de datos entrenamiento-validación para 85-15")
            st.markdown(f"<h1 style='text-align: center;'>{porc85} {porc15}</h1>", unsafe_allow_html=True)
        with col3:
            st.markdown("##### Número de datos entrenamiento-validación para 80-20")
            st.markdown(f"<h1 style='text-align: center;'>{porc80} {porc20}</h1>", unsafe_allow_html=True)
        with col4:
            st.markdown("##### Número de datos entrenamiento-validación para 75-25")
            st.markdown(f"<h1 style='text-align: center;'>{porc75} {porc25}</h1>", unsafe_allow_html=True)  
        with col5:
            st.markdown("##### Número de datos entrenamiento-validación para 70-30")
            st.markdown(f"<h1 style='text-align: center;'>{porc70} {porc30}</h1>", unsafe_allow_html=True)  
        st.markdown ("""---""")      
    with st.container():
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### Estadística básica de longitud de palabras para tweets catalogados como desastre natural")
            st.markdown(f"#### {'Media:'} {round(mean_words_1,3)}")
            st.markdown(f"#### {'Máximo:'} {round(max_words_1,3)}")
            st.markdown(f"#### {'Mínimo:'} {round(min_words_1,3)}")
            st.markdown(f"#### {'Desvío padrón:'} {round(sd_words_1,3)}")
            #st.markdown(f"<h1 style='text-align: center;'>Media {mean_words}</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown("#### Estadística básica de longitud de palabras para tweets catalogados como no desastre natural")
            st.markdown(f"#### {'Media:'} {round(mean_words_0,3)}")
            st.markdown(f"#### {'Máximo:'} {round(max_words_0,3)}")
            st.markdown(f"#### {'Mínimo:'} {round(min_words_0,3)}")
            st.markdown(f"#### {'Desvío padrón:'} {round(sd_words_0,3)}") 
with tab2:
    with st.container():
        st.markdown("##### Resultados para la proporción 90-10")
        st.dataframe(test_010_cal) 
    st.markdown ("""---""") 
    with st.container():
        st.markdown("##### Resultados para la proporción 85-15")
        st.dataframe(test_015_cal) 
    st.markdown ("""---""")
    with st.container():
        st.markdown("##### Resultados para la proporción 80-20")
        st.dataframe(test_020_cal) 
    st.markdown ("""---""") 
    with st.container():
        st.markdown("##### Resultados para la proporción 75-25")
        st.dataframe(test_025_cal) 
    st.markdown ("""---""")  
    with st.container():
        st.markdown("##### Resultados para la proporción 70-30")
        st.dataframe(test_030_cal) 
    st.markdown ("""---""")
with tab3:
    with st.container():
        st.markdown("#### Resultados para la proporción 90-10")
        st.markdown(f"##### {'Número de desastres estimados con NB:'} {len (test_010_val.loc[test_010_val['Naive Bayes']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Logistic Regression:'} {len (test_010_val.loc[test_010_val['Logistic Regression']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con SVM:'} {len (test_010_val.loc[test_010_val['SVM']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Random Forest:'} {len (test_010_val.loc[test_010_val['Random Forest']==1]) }")
        st.dataframe(test_010_val) 
    st.markdown ("""---""") 
    with st.container():
        st.markdown("#### Resultados para la proporción 85-15")
        st.markdown(f"##### {'Número de desastres estimados con NB:'} {len (test_015_val.loc[test_015_val['Naive Bayes']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Logistic Regression:'} {len (test_015_val.loc[test_015_val['Logistic Regression']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con SVM:'} {len (test_015_val.loc[test_015_val['SVM']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Random Forest:'} {len (test_015_val.loc[test_015_val['Random Forest']==1]) }")
        st.dataframe(test_015_val)
    st.markdown ("""---""") 
    with st.container():
        st.markdown("#### Resultados para la proporción 80-20")
        st.markdown(f"##### {'Número de desastres estimados con NB:'} {len (test_020_val.loc[test_020_val['Naive Bayes']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Logistic Regression:'} {len (test_020_val.loc[test_020_val['Logistic Regression']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con SVM:'} {len (test_020_val.loc[test_020_val['SVM']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Random Forest:'} {len (test_020_val.loc[test_020_val['Random Forest']==1]) }")
        st.dataframe(test_020_val)
    st.markdown ("""---""") 
    with st.container():
        st.markdown("#### Resultados para la proporción 75-25")
        st.markdown(f"##### {'Número de desastres estimados con NB:'} {len (test_025_val.loc[test_025_val['Naive Bayes']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Logistic Regression:'} {len (test_025_val.loc[test_025_val['Logistic Regression']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con SVM:'} {len (test_025_val.loc[test_025_val['SVM']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Random Forest:'} {len (test_025_val.loc[test_025_val['Random Forest']==1]) }")
        st.dataframe(test_025_val)
    st.markdown ("""---""") 
    with st.container():
        st.markdown("#### Resultados para la proporción 70-30")
        st.markdown(f"##### {'Número de desastres estimados con NB:'} {len (test_030_val.loc[test_030_val['Naive Bayes']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Logistic Regression:'} {len (test_030_val.loc[test_030_val['Logistic Regression']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con SVM:'} {len (test_030_val.loc[test_030_val['SVM']==1]) }")
        st.markdown(f"##### {'Número de desastres estimados con Random Forest:'} {len (test_030_val.loc[test_030_val['Random Forest']==1]) }")
        st.dataframe(test_030_val)
    st.markdown ("""---""") 