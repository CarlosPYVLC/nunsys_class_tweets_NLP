import streamlit as st
from PIL import Image

#st.set_page_config (page_title = 'Home', page_icon = '🗺️', layout = 'wide')

image = Image.open('CE.jpg')
st.sidebar.image (image, width=120)
st.markdown("<h1 style='text-align: justify;'>Este dashboard fue diseñado con el objetivo de facilitar la visualización de la clasificación de tweets aleatorios en relativos a desastres naturales o no</h1>", unsafe_allow_html=True)
#st.header('Datos crudos para entrenamiento en la clasificación de tweets: desatres naturales')
st.sidebar.markdown ("# NUNSYS")
st.sidebar.markdown ("## Prueba Técnica para Senior DS")
st.sidebar.markdown ("""---""")

st.markdown (
    """
    ### ¿Como seguir este dashboard?
        I. El mismo está compuesto de: 
            1. Una barra lateral con una breve explicación
            2. Una parte principal, arriba en el centro donde se exhibe el fichero tal cual se descarga de Kaggle
            3. Tres compartimientos:
                3.1. Breve análisis de los datos crudos
                3.2. Modelado de los datos para clasificar tweets y verificación del desempeño de los modelos
                3.3. Verificación de los modelos en el fichero test.csv
    ### Consultas a:
        - Carlos Echeverria (echeverriacarlospy@gmail.com)
""")

