import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import seaborn as sns

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
    # Markdown para centrar el título
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

    # Cargar dataset
    df = pd.read_csv('cnn_classification/dataset_1.csv')
    columns_to_drop = ["gender", "person_emp_exp", "credit_score", "rate_income", "loan_amount"]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    def remove_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.1)  # Primer cuartil
            Q3 = df[col].quantile(0.9)  # Tercer cuartil
            IQR = Q3 - Q1  # Rango intercuartílico
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    # Aplicar eliminación de outliers en "edad" e "income"
    df = remove_outliers(df, ["person_age", "person_income"])
    df
    # ================== Análisis Exploratorio de Datos ==================

    # Selector para elegir entre estadísticas descriptivas y gráficos
    option = st.selectbox(
        "Select the type of analysis you want to view:",
        ["📋 Descriptive Statistics", "📊 Graphs"]
    )
    
    # ================== DESCRIPTIVE STATISTICS ==================
    if option == "📋 Descriptive Statistics":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary of Data Types")
            st.write(df.dtypes)
        with col2:
            st.subheader("Missing Values per Variable")
            st.write(df.isnull().sum())

        st.subheader("Summary Statistics")
        st.write(df.describe())  # Mostrar estadísticas descriptivas

        # Selector para variables categóricas
        selected_display_name = st.selectbox("Select a categorical variable:", list(column_display_names.keys()))
        selected_col = column_display_names[selected_display_name]

        # Mostrar los valores únicos de la variable categórica seleccionada
        st.subheader(f"Value Counts for {selected_display_name}")
        st.write(df[selected_col].value_counts())

        st.subheader("Loan Approval Status Counts ✅/❌")
        st.write(df['loan_status'].value_counts())

    # ================== GRAPHICAL ANALYSIS ==================
    elif option == "📊 Graphs":
        # Selector para elegir entre variables categóricas y numéricas
        variable_type = st.radio("Select Variable Type:", ["Categorical", "Numerical"], horizontal=True)

        if variable_type == "Categorical":
            # Identificar variables categóricas
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            col1, col2 = st.columns(2)

            with col1:
                selected_display_name = st.radio("Select a categorical variable:", list(column_display_names.keys()))
                selected_col = column_display_names[selected_display_name]

            with col2:
                chart_type = st.radio("Select a chart type:", ["Pie Chart", "Bar Chart"])

            # Mostrar el gráfico seleccionado
            st.markdown(f"<h4 style='text-align: center;'>{chart_type} for {selected_display_name}</h4>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4, 3))
            if chart_type == "Pie Chart":
                df[selected_col].value_counts().plot.pie(
                    autopct='%1.1f%%', 
                    startangle=90, 
                    ax=ax, 
                    colors=plt.cm.Paired.colors,
                    textprops={'fontsize': 8}
                )
                ax.set_ylabel('')
            elif chart_type == "Bar Chart":
                df[selected_col].value_counts().plot.bar(
                    color=plt.cm.Paired.colors, 
                    edgecolor='black', 
                    ax=ax
                )
                ax.set_xlabel(selected_display_name, fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)

            st.pyplot(fig)

        elif variable_type == "Numerical":
            # Identificar variables numéricas y **remover guiones bajos** para mostrar nombres más amigables
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            numerical_display_names = {col: col.replace("_", " ").title() for col in numerical_cols}

            col1, col2 = st.columns(2)

            with col1:
                selected_display_name = st.radio("Select a numerical variable:", list(numerical_display_names.values()))
                selected_col = [col for col, display_name in numerical_display_names.items() if display_name == selected_display_name][0]

            with col2:
                chart_type = st.radio("Select a chart type:", ["Histogram", "Boxplot", "Scatter Plot"])

            # Mostrar el gráfico seleccionado
            st.markdown(f"<h4 style='text-align: center;'>{chart_type} for {selected_display_name}</h4>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4, 3))
            if chart_type == "Histogram":
                sns.histplot(df[selected_col], kde=True, ax=ax)
            elif chart_type == "Boxplot":
                sns.boxplot(x=df[selected_col], ax=ax)
            elif chart_type == "Scatter Plot":
                selected_display_name2 = st.radio("Select another numerical variable for Scatter Plot:", list(numerical_display_names.values()))
                selected_col2 = [col for col, display_name in numerical_display_names.items() if display_name == selected_display_name2][0]
                sns.scatterplot(x=df[selected_col], y=df[selected_col2], ax=ax)

            st.pyplot(fig)
    st.subheader("📈 Correlation Matrix")
    numeric_df = df.select_dtypes(include=['number'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
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
