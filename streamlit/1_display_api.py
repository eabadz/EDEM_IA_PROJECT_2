import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from PIL import Image

# ======================= CONFIGURACIÓN =======================
st.set_page_config(page_title="Predicción de Crédito", page_icon="💳", layout="wide")

# ======================= CARGA DE DATOS =======================
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")  # Asegúrate de la ruta correcta

df = load_data()
# ======================= CARGA DEL MODELO Y SCALER =======================
@st.cache_resource
def cargar_modelo():
    input_dim = 18  # Número de características después de OHE
    model = NeuralNetwork(input_dim)
    model.load_state_dict(torch.load("modelo2_credito.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def cargar_scaler():
    return joblib.load("scaler2.pkl")

# Definir la arquitectura de la red neuronal
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
        # raw_output = self.output(x)
        # print("Salida cruda del modelo:", raw_output.item())  # <-- Agrega esta línea
        # x = self.sigmoid(raw_output)

        return x

# Cargar el modelo y el escalador una sola vez
model = cargar_modelo()

scaler = cargar_scaler()

# ======================= SIDEBAR Y NAVEGACIÓN =======================
st.sidebar.title("📌 Menú")
pagina = st.sidebar.radio("Ir a:", ["Inicio", "EDA - Análisis de Datos", "Predicción", "Sobre el Modelo"])

# ======================= PÁGINA PRINCIPAL =======================
if pagina == "Inicio":
    col1, col2 = st.columns([2, 1])  

    with col1:
        st.title("🏦 Predicción de Aprobación de Crédito")
        st.markdown("## 🔍 Evalúa tu crédito en segundos")
        st.write("💡 Con esta app podrás:")
        st.write("✔ Explorar datos históricos con gráficos 📊")
        st.write("✔ Obtener predicciones en tiempo real 🔮")
        st.write("✔ Comprender qué factores influyen en la aprobación 🧠")

    with col2:
        imagen = Image.open("credit_image.jpg")
        st.image(imagen, width=250)  

# ======================= PÁGINA DE EDA =======================
elif pagina == "EDA - Análisis de Datos":
    st.title("📊 Exploración de Datos")
    st.subheader("🔍 Vista General del Dataset")
    st.write(f"El dataset tiene **{df.shape[0]} filas** y **{df.shape[1]} columnas**.")
    st.dataframe(df.head())

    # Estadísticas descriptivas
    st.subheader("📌 Estadísticas Descriptivas")
    st.write("📊 **Variables numéricas:**")
    st.write(df.describe())

    st.write("🔠 **Variables categóricas:**")
    st.write(df.describe(include='object'))

    # Visualización interactiva
    st.subheader("📊 Visualización de Datos")
    tipo_grafico = st.selectbox("Selecciona el tipo de gráfico:", 
                                ["Histograma", "Boxplot (Outliers)", "Gráfico de barras", "Scatter Plot"])
    variable_x = st.selectbox("Selecciona la variable X:", df.columns)
    variable_y = st.selectbox("Selecciona la variable Y (opcional, solo para scatter):", ["Ninguna"] + list(df.columns))

    fig, ax = plt.subplots(figsize=(8, 4))

    if tipo_grafico == "Histograma":
        sns.histplot(df[variable_x], kde=True, ax=ax)
    elif tipo_grafico == "Boxplot (Outliers)":
        sns.boxplot(x=df[variable_x], ax=ax)
    elif tipo_grafico == "Gráfico de barras":
        sns.countplot(data=df, x=variable_x, ax=ax)
        plt.xticks(rotation=45)
    elif tipo_grafico == "Scatter Plot" and variable_y != "Ninguna":
        sns.scatterplot(data=df, x=variable_x, y=variable_y, ax=ax)

    st.pyplot(fig)

    # Heatmap de correlaciones
    st.subheader("📈 Mapa de Calor de Correlaciones")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# ======================= PÁGINA DE PREDICCIÓN =======================
elif pagina == "Predicción":
    st.title("🔮 Predicción de Aprobación de Crédito")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Edad", min_value=18, max_value=100, step=1)
        person_income = st.number_input("Ingreso Anual ($)", min_value=1000, max_value=500000, step=1000)
        loan_amnt = st.number_input("Monto del Préstamo ($)", min_value=500, max_value=100000, step=500)
        # loan_int_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, max_value=50.0, step=0.1)

    with col2:
        loan_int_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, max_value=50.0, step=0.1)
        # credit_score = st.number_input("Puntaje de Crédito", min_value=300, max_value=850, step=1)
        cb_hist_length = st.number_input("Historial de Crédito (años)", min_value=0, max_value=50, step=1)
        previous_loan_defaults = st.selectbox("Préstamos impagos previos", ["No", "Sí"])

    loan_intent = st.selectbox("Motivo del Préstamo", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
    person_education = st.selectbox("Nivel de Educación", ["High School", "Bachelor", "Master", "Doctorate"])
    person_home_ownership = st.selectbox("Tipo de Vivienda", ["RENT", "OWN", "OTHER"])

    columnas_modelo = [
        'person_age', 'person_income', 'loan_amnt', 'loan_int_rate',
        'cb_person_cred_hist_length',
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
        'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'person_education_Bachelor', 'person_education_Doctorate',
        'person_education_High School', 'person_education_Master',
        'person_home_ownership_OTHER', 'person_home_ownership_OWN',
        'person_home_ownership_RENT', 'previous_loan_defaults_on_file_Yes'
    ]

    def procesar_datos():
    # Crear DataFrame con los valores ingresados
        df = pd.DataFrame([[person_age, person_income, loan_amnt, loan_int_rate, 
                          cb_hist_length, previous_loan_defaults, 
                          loan_intent, person_education, person_home_ownership]],
                        columns=['person_age', 'person_income', 'loan_amnt', 'loan_int_rate',
                                 'cb_person_cred_hist_length',
                                 'previous_loan_defaults_on_file',
                                 'loan_intent', 'person_education', 'person_home_ownership'])

    # Convertir 'previous_loan_defaults_on_file' a 0/1
        df['previous_loan_defaults_on_file'] = (df['previous_loan_defaults_on_file'] == "Sí").astype(int)

    # ======================= ONE-HOT ENCODING MANUAL =======================
    # Definir las categorías esperadas en el modelo
        categorias_loan_intent = ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
        categorias_education = ["High School", "Bachelor", "Master", "Doctorate"]
        categorias_home_ownership = ["RENT", "OWN", "OTHER"]

    # Crear columnas OHE con valores por defecto 0
        for categoria in categorias_loan_intent:
            df[f'loan_intent_{categoria}'] = int(loan_intent == categoria)

        for categoria in categorias_education:
            df[f'person_education_{categoria}'] = int(person_education == categoria)

        for categoria in categorias_home_ownership:
            df[f'person_home_ownership_{categoria}'] = int(person_home_ownership == categoria)

    # ======================= NORMALIZACIÓN =======================
    # Seleccionar las variables numéricas
        numerical_features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length']
    
    # Aplicar MinMaxScaler solo a las variables numéricas
        df[numerical_features] = scaler.transform(df[numerical_features])

    # ======================= ASEGURAR ORDEN DE COLUMNAS =======================
    # Definir el orden de las columnas según el modelo entrenado
        columnas_modelo = [
            'person_age', 'person_income', 'loan_amnt', 'loan_int_rate',
            'cb_person_cred_hist_length', 
            'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
            'loan_intent_PERSONAL', 'loan_intent_VENTURE',
            'person_education_Bachelor', 'person_education_Doctorate',
            'person_education_High School', 'person_education_Master',
            'person_home_ownership_OTHER', 'person_home_ownership_OWN',
            'person_home_ownership_RENT', 'previous_loan_defaults_on_file'
    ]

    # Asegurar que todas las columnas del modelo están presentes
        for col in columnas_modelo:
            if col not in df.columns:
                df[col] = 0  # Agregar cualquier columna faltante con valor 0

    # Ordenar las columnas en el mismo orden que el modelo
        df = df[columnas_modelo]

    # Convertir a `float` para evitar errores con PyTorch
        df = df.astype(float)

    # Convertir a tensor de PyTorch
        return torch.tensor(df.values, dtype=torch.float32)


    if st.button("🔮 Predecir"):
        datos_procesados = procesar_datos()  # Procesar los datos ingresados

        # Mostrar las columnas procesadas
        # st.write("Columnas del input en Streamlit:", list(df.columns))
        # st.write("Columnas del modelo entrenado:", columnas_modelo)

        # st.write("📊 **Datos Normalizados para el Modelo:**")
        # st.write(pd.DataFrame(datos_procesados.numpy(), columns=columnas_modelo))  

        # Realizar la predicción
        prediccion = model(datos_procesados)

        # Resultado de la predicción
        resultado = "✅ Crédito Aprobado" if prediccion.item() > 0.5 else "❌ Crédito Rechazado"
        st.subheader("📌 Resultado de la Predicción")
        st.write(resultado)
        st.write("📊 Probabilidad Predicha:", prediccion.item())

        
        










