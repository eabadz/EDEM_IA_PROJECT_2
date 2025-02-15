# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import joblib
# from PIL import Image

# # ======================= CONFIGURACIÓN =======================
# st.set_page_config(page_title="Predicción de Crédito", page_icon="💳", layout="wide")

# # ======================= CARGA DE DATOS =======================
# @st.cache_data
# def load_data():
#     return pd.read_csv("dataset.csv")  # Asegúrate de la ruta correcta

# df = load_data()

# # ======================= CARGA DEL MODELO Y SCALER =======================
# @st.cache_resource
# def cargar_modelo():
#     input_dim = 19  # Número de características después de OHE
#     model = NeuralNetwork(input_dim)
#     model.load_state_dict(torch.load("modelo_credito.pth", map_location=torch.device("cpu")))
#     model.eval()
#     return model

# @st.cache_resource
# def cargar_scaler():
#     return joblib.load("scaler.pkl")

# # Definir la arquitectura de la red neuronal
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.output = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc3(x))
#         x = self.sigmoid(self.output(x))
#         return x

# # Cargar el modelo y el escalador una sola vez
# model = cargar_modelo()
# scaler = cargar_scaler()

# # ======================= SIDEBAR Y NAVEGACIÓN =======================
# st.sidebar.title("📌 Menú")
# pagina = st.sidebar.radio("Ir a:", ["Inicio", "EDA - Análisis de Datos", "Predicción", "Sobre el Modelo"])

# # ======================= PÁGINA PRINCIPAL =======================
# if pagina == "Inicio":
#     col1, col2 = st.columns([2, 1])  

#     with col1:
#         st.title("🏦 Predicción de Aprobación de Crédito")
#         st.markdown("## 🔍 Evalúa tu crédito en segundos")
#         st.write("💡 Con esta app podrás:")
#         st.write("✔ Explorar datos históricos con gráficos 📊")
#         st.write("✔ Obtener predicciones en tiempo real 🔮")
#         st.write("✔ Comprender qué factores influyen en la aprobación 🧠")

#     with col2:
#         imagen = Image.open("credit_image.jpg")
#         st.image(imagen, width=250)  

# # ======================= PÁGINA DE EDA =======================
# elif pagina == "EDA - Análisis de Datos":
#     st.title("📊 Exploración de Datos")
#     st.subheader("🔍 Vista General del Dataset")
#     st.write(f"El dataset tiene **{df.shape[0]} filas** y **{df.shape[1]} columnas**.")
#     st.dataframe(df.head())

#     # Estadísticas descriptivas
#     st.subheader("📌 Estadísticas Descriptivas")
#     st.write("📊 **Variables numéricas:**")
#     st.write(df.describe())

#     st.write("🔠 **Variables categóricas:**")
#     st.write(df.describe(include='object'))

#     # Visualización interactiva
#     st.subheader("📊 Visualización de Datos")
#     tipo_grafico = st.selectbox("Selecciona el tipo de gráfico:", 
#                                 ["Histograma", "Boxplot (Outliers)", "Gráfico de barras", "Scatter Plot"])
#     variable_x = st.selectbox("Selecciona la variable X:", df.columns)
#     variable_y = st.selectbox("Selecciona la variable Y (opcional, solo para scatter):", ["Ninguna"] + list(df.columns))

#     fig, ax = plt.subplots(figsize=(8, 4))

#     if tipo_grafico == "Histograma":
#         sns.histplot(df[variable_x], kde=True, ax=ax)
#     elif tipo_grafico == "Boxplot (Outliers)":
#         sns.boxplot(x=df[variable_x], ax=ax)
#     elif tipo_grafico == "Gráfico de barras":
#         sns.countplot(data=df, x=variable_x, ax=ax)
#         plt.xticks(rotation=45)
#     elif tipo_grafico == "Scatter Plot" and variable_y != "Ninguna":
#         sns.scatterplot(data=df, x=variable_x, y=variable_y, ax=ax)

#     st.pyplot(fig)

#     # Heatmap de correlaciones
#     st.subheader("📈 Mapa de Calor de Correlaciones")
#     numeric_df = df.select_dtypes(include=['number'])
#     fig, ax = plt.subplots(figsize=(10,6))
#     sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
#     st.pyplot(fig)


# # ======================= PÁGINA DE PREDICCIÓN =======================
# if pagina == "Predicción":
#     st.title("🔮 Predicción de Aprobación de Crédito")

#     # Formulario de entrada (sin la columna `loan_status` que no se utiliza)
#     person_age = st.number_input("Edad", min_value=18, max_value=100, step=1)
#     person_income = st.number_input("Ingreso Anual ($)", min_value=1000, max_value=500000, step=1000)
#     loan_amnt = st.number_input("Monto del Préstamo ($)", min_value=500, max_value=100000, step=500)
#     loan_int_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, max_value=50.0, step=0.1)
#     cb_hist_length = st.number_input("Historial de Crédito (años)", min_value=0, max_value=50, step=1)
#     credit_score = st.number_input("Puntaje de Crédito", min_value=300, max_value=850, step=1)
#     previous_loan_defaults = st.selectbox("Préstamos impagos previos", ["No", "Sí"])

#     # 1. One-Hot Encoding de `person_education`
#     person_education_Bachelor = 1 if st.selectbox("¿Nivel de educación?", ["Bachillerato", "Licenciatura", "Máster", "Doctorado"]) == "Licenciatura" else 0
#     person_education_Doctorate = 1 if st.selectbox("¿Nivel de educación?", ["Bachillerato", "Licenciatura", "Máster", "Doctorado"]) == "Doctorado" else 0
#     person_education_High_School = 1 if st.selectbox("¿Nivel de educación?", ["Bachillerato", "Licenciatura", "Máster", "Doctorado"]) == "Bachillerato" else 0
#     person_education_Master = 1 if st.selectbox("¿Nivel de educación?", ["Bachillerato", "Licenciatura", "Máster", "Doctorado"]) == "Máster" else 0

#     # 2. One-Hot Encoding de `person_home_ownership`
#     person_home_ownership_OWN = 1 if st.selectbox("¿Eres propietario de tu vivienda?", ["Sí", "No"]) == "Sí" else 0
#     person_home_ownership_RENT = 1 if st.selectbox("¿Eres arrendatario de tu vivienda?", ["Sí", "No"]) == "Sí" else 0
#     person_home_ownership_OTHER = 1 if st.selectbox("¿Otra forma de propiedad?", ["Sí", "No"]) == "Sí" else 0

#     # 3. One-Hot Encoding de `loan_intent`
#     loan_intent_EDUCATION = 1 if st.selectbox("¿Propósito del préstamo?", ["Educación", "Mejoras del hogar", "Médico", "Personal", "Emprendimiento"]) == "Educación" else 0
#     loan_intent_HOMEIMPROVEMENT = 1 if st.selectbox("¿Propósito del préstamo?", ["Educación", "Mejoras del hogar", "Médico", "Personal", "Emprendimiento"]) == "Mejoras del hogar" else 0
#     loan_intent_MEDICAL = 1 if st.selectbox("¿Propósito del préstamo?", ["Educación", "Mejoras del hogar", "Médico", "Personal", "Emprendimiento"]) == "Médico" else 0
#     loan_intent_PERSONAL = 1 if st.selectbox("¿Propósito del préstamo?", ["Educación", "Mejoras del hogar", "Médico", "Personal", "Emprendimiento"]) == "Personal" else 0
#     loan_intent_VENTURE = 1 if st.selectbox("¿Propósito del préstamo?", ["Educación", "Mejoras del hogar", "Médico", "Personal", "Emprendimiento"]) == "Emprendimiento" else 0

#     # 4. One-Hot Encoding de `previous_loan_defaults_on_file`
#     previous_loan_defaults_on_file_Yes = 1 if previous_loan_defaults == "Sí" else 0

#     # Crear el vector de entrada para la predicción (sin `loan_status`)
#     entrada = [
#         person_age,
#         person_income,
#         loan_amnt,
#         loan_int_rate,
#         cb_hist_length,
#         credit_score,
#         loan_intent_EDUCATION,
#         loan_intent_HOMEIMPROVEMENT,
#         loan_intent_MEDICAL,
#         loan_intent_PERSONAL,
#         loan_intent_VENTURE,
#         person_education_Bachelor,
#         person_education_Doctorate,
#         person_education_High_School,
#         person_education_Master,
#         person_home_ownership_OTHER,
#         person_home_ownership_OWN,
#         person_home_ownership_RENT,
#         previous_loan_defaults_on_file_Yes
#     ]

#     # 5. Escalado de las variables numéricas
#     entrada_scaled = scaler.transform([entrada[:6]])  # Solo las primeras 6 son numéricas
#     entrada_final = entrada_scaled[0].tolist() + entrada[6:]  # Concatenar con las variables categóricas

#     # 6. Convertir a tensor y realizar la predicción
#     datos_procesados = torch.tensor([entrada_final], dtype=torch.float32)
#     prediccion = model(datos_procesados)
#     resultado = "✅ Crédito Aprobado" if prediccion.item() > 0.5 else "❌ Crédito Rechazado"
    
#     st.subheader("📌 Resultado de la Predicción")
#     st.write(resultado)
#     st.write("📊 Probabilidad Predicha:", round(prediccion.item(), 4))


import torch
import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import numpy as np

# Cargar el modelo y el escalador
@st.cache_resource
def cargar_modelo():
    input_dim = 19  # Número de características después de OHE
    model = NeuralNetwork(input_dim)
    model.load_state_dict(torch.load("modelo_credito.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def cargar_scaler():
    return joblib.load("scaler.pkl")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.output(x))
        return x

# Cargar el modelo y el escalador una sola vez
model = cargar_modelo()
scaler = cargar_scaler()

# Función para aplicar el preprocesamiento adecuado
def preprocesar_datos(person_age, person_income, loan_amnt, loan_int_rate, cb_hist_length, credit_score,
                      loan_intent, person_education, person_home_ownership, previous_loan_defaults):
    # Codificación One-Hot de las variables categóricas
    loan_intent_EDUCATION = 1 if loan_intent == "EDUCATION" else 0
    loan_intent_HOMEIMPROVEMENT = 1 if loan_intent == "HOMEIMPROVEMENT" else 0
    loan_intent_MEDICAL = 1 if loan_intent == "MEDICAL" else 0
    loan_intent_PERSONAL = 1 if loan_intent == "PERSONAL" else 0
    loan_intent_VENTURE = 1 if loan_intent == "VENTURE" else 0

    person_education_Bachelor = 1 if person_education == "Bachelor" else 0
    person_education_Doctorate = 1 if person_education == "Doctorate" else 0
    person_education_High_School = 1 if person_education == "High School" else 0
    person_education_Master = 1 if person_education == "Master" else 0

    person_home_ownership_OWN = 1 if person_home_ownership == "OWN" else 0
    person_home_ownership_RENT = 1 if person_home_ownership == "RENT" else 0
    person_home_ownership_OTHER = 1 if person_home_ownership == "OTHER" else 0

    previous_loan_defaults_on_file_Yes = 1 if previous_loan_defaults == "Yes" else 0

    # Crear la entrada de datos para el modelo
    entrada = [
        person_age,
        person_income,
        loan_amnt,
        loan_int_rate,
        cb_hist_length,
        credit_score,
        loan_intent_EDUCATION,
        loan_intent_HOMEIMPROVEMENT,
        loan_intent_MEDICAL,
        loan_intent_PERSONAL,
        loan_intent_VENTURE,
        person_education_Bachelor,
        person_education_Doctorate,
        person_education_High_School,
        person_education_Master,
        person_home_ownership_OTHER,
        person_home_ownership_OWN,
        person_home_ownership_RENT,
        previous_loan_defaults_on_file_Yes
    ]

    # Escalar las primeras 6 variables numéricas
    entrada_scaled = scaler.transform([entrada[:6]])  # Solo las primeras 6 son numéricas
    entrada_final = entrada_scaled[0].tolist() + entrada[6:]  # Concatenar con las variables categóricas

    return entrada_final

# Función para hacer la predicción
def hacer_prediccion(person_age, person_income, loan_amnt, loan_int_rate, cb_hist_length, credit_score,
                     loan_intent, person_education, person_home_ownership, previous_loan_defaults):
    # Preprocesar los datos
    entrada_procesada = preprocesar_datos(person_age, person_income, loan_amnt, loan_int_rate, cb_hist_length, credit_score,
                                          loan_intent, person_education, person_home_ownership, previous_loan_defaults)
    
    # Convertir la entrada a tensor
    datos_procesados = torch.tensor([entrada_procesada], dtype=torch.float32)

    # Realizar la predicción
    with torch.no_grad():
        prediccion = model(datos_procesados)
    
    # Obtener la clase predicha y la probabilidad
    clase_predicha = "✅ Crédito Aprobado" if prediccion.item() > 0.5 else "❌ Crédito Rechazado"
    probabilidad_prediccion = prediccion.item()

    return clase_predicha, probabilidad_prediccion

# Formulario de entrada
person_age = st.number_input("Edad", min_value=18, max_value=100, step=1)
person_income = st.number_input("Ingreso Anual ($)", min_value=1000, max_value=500000, step=1000)
loan_amnt = st.number_input("Monto del Préstamo ($)", min_value=500, max_value=100000, step=500)
loan_int_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, max_value=50.0, step=0.1)
cb_hist_length = st.number_input("Historial de Crédito (años)", min_value=0, max_value=50, step=1)
credit_score = st.number_input("Puntaje de Crédito", min_value=300, max_value=850, step=1)
previous_loan_defaults = st.selectbox("Préstamos impagos previos", ["No", "Sí"])

# One-Hot Encoding para las otras variables
person_education = st.selectbox("Nivel de Educación", ["Bachillerato", "Licenciatura", "Máster", "Doctorado"])
person_home_ownership = st.selectbox("¿Eres propietario de tu vivienda?", ["Sí", "No"])
loan_intent = st.selectbox("¿Propósito del préstamo?", ["Educación", "Mejoras del hogar", "Médico", "Personal", "Emprendimiento"])

# Realizar la predicción
resultado, probabilidad = hacer_prediccion(person_age, person_income, loan_amnt, loan_int_rate, cb_hist_length, credit_score,
                                           loan_intent, person_education, person_home_ownership, previous_loan_defaults)

# Mostrar el resultado
st.subheader("📌 Resultado de la Predicción")
st.write(resultado)
st.write("📊 Probabilidad Predicha:", round(probabilidad, 4))















