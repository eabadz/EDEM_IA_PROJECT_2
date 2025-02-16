import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from PIL import Image
import sqlite3
from datetime import datetime

# ======================= CONFIGURACIÓN =======================
st.set_page_config(page_title="Predicción de Crédito", page_icon="💳", layout="wide")

# ======================= CARGA DE DATOS =======================
@st.cache_data
def load_data():
    return pd.read_csv("streamlit/dataset.csv")  # Asegúrate de la ruta correcta

df = load_data()

# ======================= CREAR LA CONEXION A LA BASE DE DATOS =======================


def create_connection(db_file):
    """Crea una conexión a la base de datos SQLite especificada."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print("Conexión establecida con la BD")
    except Exception as e:
        print(e)
    return conn

# ======================= CREAR LA BASE DE DATOS =======================
def crear_base_datos():
    """Crea la base de datos y la tabla si no existen"""
    conn = sqlite3.connect(r"streamlit\loan_predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS loan_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_age INTEGER,
            person_income REAL,
            person_education TEXT,
            person_home_ownership TEXT,
            loan_amount REAL,
            loan_intent TEXT,
            loan_int_rate REAL,
            cb_person_cred_hist_length INTEGER,
            previous_loan_defaults_on_file TEXT,
            prediction INTEGER,
            date_time TEXT
        )
    """)
    conn.commit()
    conn.close()
crear_base_datos()

# ======================= FUNCION PARA INSERTAR LOS DATOS EN LA BASE DE DATOS =======================
def fetch_predictions():
    conn = sqlite3.connect("streamlit\loan_predictions.db")
    query = "SELECT * FROM loan_predictions ORDER BY date_time DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Eliminar índices duplicados si existen
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    return df

# ======================= CREAR LA ARQUITECTURA DE LA RED NEURONAL =======================

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
    
# ======================= CARGA DEL MODELO Y SCALER =======================
@st.cache_resource
def cargar_modelo():
    input_dim = 18  # Número de características después de OHE
    model = NeuralNetwork(input_dim)
    model.load_state_dict(torch.load("streamlit\modelo2_credito.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def cargar_scaler():
    return joblib.load("streamlit\scaler2.pkl")

# Cargar el modelo y el escalador una sola vez
model = cargar_modelo()

scaler = cargar_scaler()

# ======================= SIDEBAR Y NAVEGACIÓN =======================
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #00569D; /* Azul oscuro, puedes ajustar el tono */
        color: white; /* Establecer el color de las letras en blanco */
    }
    [data-testid="stSidebar"] .css-1d391kg {
        color: white; /* También aseguramos que el título sea blanco */
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Menú")
pagina = st.sidebar.radio("Ir a:", ["Inicio", "EDA - Análisis de Datos", "Predicción", "Login", "About Us"])

# ======================= PÁGINA PRINCIPAL =======================
if pagina == "Inicio":
    col1, col2 = st.columns([2, 1])  

    with col1:
        st.title(" Predicción de Aprobación de Crédito")
        st.markdown("---")
      
     # Sección de introducción con estilo mejorado
    st.markdown("##  Introducción", unsafe_allow_html=True)
    st.write(
        "<div style='text-align: justify; font-size: 16px;'>"
        "Esta aplicación utiliza un modelo de machine learning para predecir la aprobación de créditos bancarios. "
        "A través del análisis de diversas características financieras y personales, el modelo puede estimar si una solicitud "
        "de crédito tiene una probabilidad de ser aprobada o no. Esta herramienta es útil tanto "
        "para individuos que deseen entender mejor los factores que influyen en una decisión crediticia."
        "</div>", unsafe_allow_html=True
    )

    # Explicación de las variables con estilo mejorado
    st.markdown("## 📋 Loan Dataset Features")
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    features = {
        "person_age": "Edad de la persona",
        "person_education": "Nivel educativo más alto alcanzado",
        "person_income": "Ingreso anual en dólares",
        "person_home_ownership": "Estado de propiedad de la vivienda",
        "loan_amnt": "Monto solicitado del préstamo",
        "loan_intent": "Propósito del préstamo",
        "loan_int_rate": "Tasa de interés del préstamo",
        "cb_person_cred_hist_length": "Duración del historial crediticio en años",
        "previous_loan_defaults_on_file": "Indicador de impagos previos",
        "loan_status": "Estado del préstamo: 1 = aprobado, 0 = rechazado"
    }

    for feature, description in features.items():
        st.markdown(f"<div style='color: #4CAF50; font-weight: bold;'>{feature}</div> <div style='margin-bottom: 10px;'>{description}</div>", unsafe_allow_html=True)

    st.markdown("## 🎯 ¿Qué buscamos con este modelo?")
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
    st.write("🔍 *Explorar* datos históricos con gráficos interactivos ")
    st.write("⚡ *Obtener* predicciones en tiempo real ")
    st.write("🧠 *Comprender* qué factores influyen en la aprobación ")

    with col2:
        imagen = Image.open(r"IMAGENES\logo_png.png")
        st.image(imagen, width=250)
        st.markdown("<p style='text-align: center; font-size: 12px;'></p>", unsafe_allow_html=True)

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
    # Título de la página
    st.title("🔮 Predicción de Aprobación de Crédito")

    # ==================== 📌 DEFINICIÓN DE LA BASE DE DATOS ==================== #
    def insert_prediction(data):
        """Inserta una nueva predicción en la base de datos."""
        conn = sqlite3.connect("loan_predictions.db")
        cursor = conn.cursor()
        try:
            sql_insert = """
            INSERT INTO loan_predictions (
                person_age, person_income, 
                person_education, person_home_ownership, loan_amount, 
                loan_intent, loan_int_rate, 
                cb_person_cred_hist_length, previous_loan_defaults_on_file, 
                prediction, date_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(sql_insert, data)
            conn.commit()
        except Exception as e:
            print(f"Error al insertar la predicción: {e}")
        finally:
            conn.close()
    # ==================== 📌 CAPTURA DE DATOS EN STREAMLIT ==================== #
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Edad", min_value=18, max_value=100, step=1)
        person_income = st.number_input("Ingreso Anual (€)", min_value=1000, max_value=10000000, step=1000)
        loan_amnt = st.number_input("Monto del Préstamo (€)", min_value=500, max_value=1000000, step=500)

    with col2:
        loan_int_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, max_value=50.0, step=0.1)
        cb_hist_length = st.number_input("Historial de Crédito (años)", min_value=0, max_value=50, step=1)
        previous_loan_defaults = st.selectbox("Préstamos impagos previos", ["No", "Sí"])

    loan_intent = st.selectbox("Motivo del Préstamo", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
    person_education = st.selectbox("Nivel de Educación", ["High School", "Bachelor", "Master", "Doctorate"])
    person_home_ownership = st.selectbox("Tipo de Vivienda", ["RENT", "OWN", "OTHER"])

    # ==================== 📌 PROCESAR LOS DATOS ==================== #
    def procesar_datos():
        """Convierte la entrada del usuario en un formato compatible con el modelo."""
        df = pd.DataFrame([[person_age, person_income, loan_amnt, loan_int_rate, 
                            cb_hist_length, previous_loan_defaults, 
                            loan_intent, person_education, person_home_ownership]],
                        columns=['person_age', 'person_income', 'loan_amnt', 'loan_int_rate',
                                'cb_person_cred_hist_length',
                                'previous_loan_defaults_on_file',
                                'loan_intent', 'person_education', 'person_home_ownership'])

        # Convertir 'previous_loan_defaults_on_file' a 0/1
        df['previous_loan_defaults_on_file'] = (df['previous_loan_defaults_on_file'] == "Sí").astype(int)

        # ======================= ONE-HOT ENCODING =======================
        categorias_loan_intent = ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
        categorias_education = ["High School", "Bachelor", "Master", "Doctorate"]
        categorias_home_ownership = ["RENT", "OWN", "OTHER"]

        for categoria in categorias_loan_intent:
            df[f'loan_intent_{categoria}'] = int(loan_intent == categoria)

        for categoria in categorias_education:
            df[f'person_education_{categoria}'] = int(person_education == categoria)

        for categoria in categorias_home_ownership:
            df[f'person_home_ownership_{categoria}'] = int(person_home_ownership == categoria)

        # ======================= NORMALIZACIÓN =======================
        numerical_features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length']
        df[numerical_features] = scaler.transform(df[numerical_features])

        # ======================= ORDEN DE COLUMNAS =======================
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

        for col in columnas_modelo:
            if col not in df.columns:
                df[col] = 0  # Agregar cualquier columna faltante con valor 0

        df = df[columnas_modelo]
        df = df.astype(float)

        return torch.tensor(df.values, dtype=torch.float32)

    # ==================== 📌 BOTÓN DE PREDICCIÓN ==================== #
    if st.button(" PREDECIR"):
        datos_procesados = procesar_datos()  # Procesar los datos ingresados
        prediccion = model(datos_procesados)
        resultado = 1 if prediccion.item() > 0.5 else 0  # Convertir a 0 o 1
        
        # Mostrar el resultado
        st.subheader("📌 Resultado de la Predicción")
        st.write("✅ Crédito Aprobado" if resultado == 1 else "❌ Crédito Rechazado")
        st.write("📊 Probabilidad Predicha:", prediccion.item())

        # ==================== 📌 GUARDAR EN LA BASE DE DATOS ==================== #
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        insert_prediction((
            person_age, person_income, cb_hist_length,  # Person Info
            person_education, person_home_ownership,  # Education & Home Ownership
            loan_amnt, loan_intent, loan_int_rate,  # Loan Info  
            previous_loan_defaults, resultado, date_time  # Default history & prediction
        ))
    #==================== 📌 INICIALIZAR LOS CAMPOS EN session_state ==================== #

elif pagina == "Login":
    # Inicializar sesión si no existe
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Si no está autenticado, mostrar la pantalla de login
    if not st.session_state["authenticated"]:
        st.title("🔑 Login")

        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")

        if st.button("LOGIN"):
            if username == "admin" and password == "1234":
                st.session_state["authenticated"] = True
                st.rerun()  # 🔄 Recargar la página para mostrar "Database"
            else:
                st.error("❌ Invalid username or password.")

    # Si ya está autenticado, mostrar la página "Database"
    if st.session_state["authenticated"]:
        st.title("📂 Historial de Predicciones")
        st.write("")
        # Botón de logout
        

        # Obtener los datos de la base de datos
        df = fetch_predictions()

        # Si no hay datos, mostrar mensaje
        if df.empty:
            st.warning("⚠️ No hay predicciones almacenadas en la base de datos.")
            st.stop()

        # Mostrar la tabla de datos filtrada
        st.dataframe(df.reset_index(drop=True))
        if st.button("🔓 Logout"):
            st.session_state["authenticated"] = False
            st.rerun()
elif pagina == "About Us":  
    st.title("👥 About Us")
    st.write("Conoce al equipo detrás de este proyecto.")
    team_members = [
    {"name": "Jose Barbero", "linkedin": "https://www.linkedin.com/in/jose-barbero-bru-7168a3214/"},
    {"name": "Gonzálo López", "linkedin": "https://www.linkedin.com/in/gonzalolopezblanquer/"},
    {"name": "Jaime Olano", "linkedin": "https://www.linkedin.com/in/jaime-olano-lopez-9a2199270/"},
    {"name": "Eduardo Abad", "linkedin": "https://www.linkedin.com/in/eduardo-abad-zabala/"}
]

    # Mostrar la información de cada miembro
    # Crear dos columnas para mostrar los miembros
    col1, col2 = st.columns(2)

    # Dividir a los miembros entre las dos columnas
    for i, member in enumerate(team_members):
        with col1 if i % 2 == 0 else col2:
            st.subheader(member["name"])
            st.markdown(f"[🔗 LinkedIn Profile]({member['linkedin']})")
            st.write("---")  # Línea divisoria entre miembros

    st.image(r"IMAGENES\foto_grupo.jpeg")