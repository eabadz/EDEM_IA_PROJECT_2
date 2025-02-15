import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Cargar el dataframe desde el archivo CSV
df = pd.read_csv('C:/Users/pepeb/Documents/EDEM_IA_MASTER/EDEM_IA_PROJECT_2/dataset.csv')

# Barra lateral para navegación
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Selecciona una sección", ["Introducción", "Análisis EDA", "Modelo"])

# Introducción
if opcion == "Introducción":
    st.image('C:/Users/pepeb/Documents/EDEM_IA_MASTER/EDEM_IA_PROJECT_2/Gemini_Generated_Image_l05jbjl05jbjl05j.jpeg')
    st.title("Introducción")
    st.write("""
    Buenas tardes a todo el mundo, somos el grupo numero 1 y os vamos a presentar nuestro modelo de prediccion de credito bancario.
    """)
    st.write("Edu Abad")

# Análisis EDA
elif opcion == "Análisis EDA":
    st.title("Análisis EDA")
    
    # Mostrar la cabecera (head) y cola (tail) del DataFrame
    st.subheader('Vista preliminar del DataFrame')
    st.write('Primeras 5 filas del DataFrame:')
    st.write(df.head())  # Muestra las primeras filas
    st.write('Últimas 5 filas del DataFrame:')
    st.write(df.tail())  # Muestra las últimas filas

    # Mostrar estadísticas descriptivas y tipos de datos de las columnas
    st.subheader('Estadísticas descriptivas y tipos de datos')
    st.write(df.describe())  # Estadísticas descriptivas
    st.write(df.dtypes)  # Tipos de datos
    
    # Crear lista de columnas
    columnas = df.columns.tolist()

    # Crear un selector para variables categóricas y numéricas
    variable_tipo = st.selectbox('Selecciona el tipo de variable', ['Categóricas', 'Numéricas'], key='tipo_variable')

    if variable_tipo == 'Categóricas':
        # Filtrar las columnas categóricas
        columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if columnas_categoricas:
            variable_categorica = st.selectbox('Selecciona una variable categórica', columnas_categoricas)
            # Crear gráfico de barras
            st.subheader(f'Gráfico de barras de {variable_categorica}')
            plt.figure(figsize=(8, 6))
            sns.countplot(x=df[variable_categorica])
            st.pyplot()
        else:
            st.write('No hay variables categóricas en el DataFrame.')

    elif variable_tipo == 'Numéricas':
        # Filtrar las columnas numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if columnas_numericas:
            variable_numerica = st.selectbox('Selecciona una variable numérica', columnas_numericas)
            # Crear gráfico boxplot
            st.subheader(f'Boxplot de {variable_numerica}')
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=df[variable_numerica])
            st.pyplot()
        else:
            st.write('No hay variables numéricas en el DataFrame.')

    # Mostrar correlaciones entre variables numéricas
    numeric_df = df.select_dtypes(include=[np.number])  # Seleccionar solo columnas numéricas
    

    # Mostrar matriz de correlación
    st.subheader('Matriz de correlación')
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot()


# Modelo
elif opcion == "Modelo":
    st.title("Modelo")
    st.write("""
    En esta sección podrás trabajar en la construcción de modelos predictivos usando los datos de la aplicación.
    Puedes cargar el conjunto de datos y experimentar con diferentes algoritmos de Machine Learning.
    """)
    # Puedes agregar código aquí para los modelos si deseas, como la carga de un modelo de ML

# To Improve speed and cache data
@st.cache(persist=True)
def explore_data(dataset):
    df = pd.read_csv(os.path.join(dataset))
    return df
