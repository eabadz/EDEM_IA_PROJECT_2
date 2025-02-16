import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Título principal de la aplicación
# st.title("📈 LOAN APPROVAL PREDICTION")

# Mapping of display names to actual column names
column_display_names = {
            "Gender": "person_gender",
            "Education Level": "person_education",
            "Home Ownership": "person_home_ownership",
            "Loan Purpose": "loan_intent",
            "Previous Loan Defaults": "previous_loan_defaults_on_file"
        }
# Inicializar la sesión de autenticación si no existe
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
# Barra lateral para la navegación entre páginas
page = st.sidebar.radio("Menu", ["Introduction", "EDA", "Model Preditcion", "Login"])

# Página: Introducción
if page == "Introducción":
    st.header("Introducción")

# Página: Análisis Exploratorio de Datos (EDA)
elif page == "EDA":
    # Markdown to centre the header
    st.markdown("""
    <style>
    .centered-header {
        text-align: center;
        font-size: 36px;
        color: #4CAF50;
    }
    </style>
    <h2 class='centered-header'>Exploratory Data Analysis</h2>
    """, unsafe_allow_html=True)
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    df 
    # Select Box for selecting the way data is displayed
    option = st.selectbox(
    "Select the type of analysis you want to view:",
    ["📋 Descriptive Statistics", "📊 Graphs"]
)
    # Display content based on the selected option
    if option == "📋 Descriptive Statistics":
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary of Data Types")
            st.write(df.dtypes)
        with col2:
            st.subheader("Missing Values per Variable")
            st.write(df.isnull().sum())
            
        st.subheader(" Summary Statistics",)
        st.write("")
        st.write(df.describe())  # Display summary statistics like mean, median, etc.
        st.write("")

        # Selectbox showing the display names
        selected_display_name = st.selectbox("Select a categorical variable:", list(column_display_names.keys()))

        # Get the actual column name from the display name
        selected_col = column_display_names[selected_display_name]

        # Display the count of each category in the selected column
        st.subheader(f"Value Counts for {selected_display_name}")
        st.write(df[selected_col].value_counts())
        st.write("")
        st.subheader("Loan Approval Status Counts ✅/❌")
        st.write(df['loan_status'].value_counts())
    elif option == "📊 Graphs":
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Create two columns: one for selecting the categorical variable, one for selecting the chart type
        col1, col2 = st.columns(2)

        with col1:
            selected_display_name = st.radio("Select a categorical variable:", list(column_display_names.keys()))
            selected_col = column_display_names[selected_display_name]

        with col2:
            chart_type = st.radio("Select a chart type:", ["Pie Chart", "Bar Chart"])

        # Display the selected chart in a smaller container
        with st.container():
            st.markdown(f"<h4 style='text-align: center;'>{chart_type} for {selected_display_name}</h4>", unsafe_allow_html=True)

            if chart_type == "Pie Chart":
                fig, ax = plt.subplots(figsize=(3, 3))  # Smaller pie chart size
                df[selected_col].value_counts().plot.pie(
                    autopct='%1.1f%%', 
                    startangle=90, 
                    ax=ax, 
                    colors=plt.cm.Paired.colors,
                    textprops={'fontsize': 8}  # Smaller text inside the pie chart
                )
                ax.set_ylabel('')  # Hide the y-axis label for better visualization
                st.pyplot(fig)

            elif chart_type == "Bar Chart":
                fig, ax = plt.subplots(figsize=(4, 3))  # Smaller bar chart size
                df[selected_col].value_counts().plot.bar(
                    color=plt.cm.Paired.colors, 
                    edgecolor='black', 
                    ax=ax
                )
                ax.set_xlabel(selected_display_name, fontsize=10)  # Smaller axis labels
                ax.set_ylabel("Count", fontsize=10)
                plt.xticks(fontsize=8)  # Smaller tick labels
                plt.yticks(fontsize=8)
                st.pyplot(fig)

# Página: Predicción del Modelo
elif page == "Predicción del Modelo":
    st.header("Predicción del Modelo")

# Página de inicio de sesion
elif page == "Login":


    # Inicializar sesión si no existe
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Si no está autenticado, mostrar la pantalla de login
    if not st.session_state["authenticated"]:
        st.title("🔑 Login")

        email = st.text_input("📧 Email ID")
        password = st.text_input("🔒 Password", type="password")

        if st.button("LOGIN"):
            if email == "admin@example.com" and password == "password123":
                st.session_state["authenticated"] = True
                st.rerun()  # 🔄 Recargar la página para mostrar "database"

            else:
                st.error("❌ Invalid email or password.")

    # Si ya está autenticado, mostrar la página "database"
    if st.session_state["authenticated"]:
        st.title("📂 Database Page")
        st.write("Bienvenido a la base de datos. Aquí se mostrarán los registros almacenados.")
