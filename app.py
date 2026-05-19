import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción Instacart - ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo de los gráficos
sns.set_theme(style="whitegrid")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Instacart_logo.svg", width=200)
st.sidebar.header("Navegación")
opcion = st.sidebar.radio("Selecciona una sección:", ["Inicio y Análisis Exploratorio (EDA)", "Predicción con Modelos", "Conclusiones"])

st.sidebar.markdown("---")
st.sidebar.write("🔗 **Recursos Externos**")
# Enlace solicitado a Google Colab
st.sidebar.link_button("Ver Notebook en Google Colab", "https://colab.research.google.com/drive/1IT-1KWfAL5K_Ur0qMz9-WeWZZvzsMuE4#scrollTo=ZuQ26sKpGTod")

# Título Principal
st.title("📊 Análisis y Predicción de Canasta de Compra")
st.markdown("---")

# Simulación de datos basada en el muestreo de tu archivo .ipynb
@st.cache_data
def cargar_datos_locales():
    np.random.seed(42)
    n_samples = 20208
    data = {
        'add_to_cart_order': np.random.randint(1, 15, size=n_samples),
        'reordered': np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
        'order_dow': np.random.randint(0, 7, size=n_samples),
        'order_hour_of_day': np.random.normal(13.5, 4.2, n_samples).astype(int).clip(0, 23),
        'days_since_prior_order': np.random.uniform(0, 30, size=n_samples),
        'department_id': np.random.randint(1, 22, size=n_samples)
    }
    return pd.DataFrame(data)

df = cargar_datos_locales()

# --- SECCIÓN 1: INICIO Y EDA ---
if opcion == "Inicio y Análisis Exploratorio (EDA)":
    st.header("📝 Introducción al Proyecto")
    
    col_intro1, col_intro2 = st.columns([2, 1])
    
    with col_intro1:
        st.markdown(f"""
        ### El Dataset: Instacart Market Basket Analysis
        Este aplicativo utiliza datos de **Instacart**, una plataforma líder en entrega de comestibles. El conjunto de datos original es una base de datos relacional que contiene más de 3 millones de pedidos de comestibles de más de 200,000 usuarios de Instacart.
        
        **Objetivo del análisis:**
        El objetivo es comprender el comportamiento de compra del usuario para predecir a qué categoría o **departamento** pertenecen los productos basándonos en hábitos temporales (hora, día) y de logística del carrito (orden de inserción).
        
        **Características utilizadas:**
        - **add_to_cart_order:** Orden en el que se añadió el producto al carrito.
        - **reordered:** Indica si el usuario ya había comprado este producto antes.
        - **order_dow:** Día de la semana (0-6).
        - **order_hour_of_day:** Hora del día en que se hizo el pedido (0-23).
        - **days_since_prior_order:** Días transcurridos desde la última compra.
        """)
    
    with col_intro2:
        st.info("💡 **Acceso al Código:** Puedes revisar todo el proceso de limpieza, entrenamiento de modelos y exportación en el siguiente enlace:")
        st.link_button("🚀 Abrir en Google Colab", "https://colab.research.google.com/drive/1IT-1KWfAL5K_Ur0qMz9-WeWZZvzsMuE4#scrollTo=ZuQ26sKpGTod")

    st.markdown("---")
    st.header("📈 Análisis Exploratorio de Datos (EDA)")
    
    # Métricas Clave
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros en la Muestra", f"{df.shape[0]}")
    col2.metric("Promedio Hora de Compra", "13.5 (1:30 PM)")
    col3.metric("Clases Principales", "Top 5 Deptos")
    
    st.markdown("### Distribución de Pedidos según la Hora del Día")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x='order_hour_of_day', palette='viridis', ax=ax)
    plt.title('Concentración de pedidos por hora')
    st.pyplot(fig)

# --- SECCIÓN 2: PREDICCIÓN ---
elif opcion == "Predicción con Modelos":
    st.header("🔮 Predicción del Departamento")
    st.write("Selecciona un modelo y configura los parámetros para predecir el `department_id` correspondiente.")

    st.sidebar.header("Modelo Predictivo")
    modelo_seleccionado = st.sidebar.selectbox(
        "Escoge el modelo:",
        ["Random Forest", "Hist-Gradient Boosting"]
    )

    st.subheader("Configuración del Pedido")
    c1, c2 = st.columns(2)
    
    with c1:
        add_to_cart_order = st.number_input("Posición en el carrito:", min_value=1, max_value=50, value=5)
        reordered = st.selectbox("¿Es re-compra?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        order_dow = st.slider("Día de la semana (0=Dom, 6=Sáb):", min_value=0, max_value=6, value=1)

    with c2:
        order_hour_of_day = st.slider("Hora del día:", min_value=0, max_value=23, value=12)
        days_since_prior_order = st.number_input("Días desde el último pedido:", min_value=0.0, max_value=30.0, value=7.0)

    nombre_archivo = "modelo_random_forest.pkl" if modelo_seleccionado == "Random Forest" else "modelo_gradient_boosting.pkl"
    ruta_modelo = os.path.join("modelos", nombre_archivo)

    if st.button("🚀 Calcular Predicción"):
        if os.path.exists(ruta_modelo):
            try:
                modelo = joblib.load(ruta_modelo)
                features = pd.DataFrame([{
                    'add_to_cart_order': add_to_cart_order,
                    'reordered': reordered,
                    'order_dow': order_dow,
                    'order_hour_of_day': order_hour_of_day,
                    'days_since_prior_order': days_since_prior_order
                }])
                
                prediccion = modelo.predict(features)[0]
                st.success(f"### El producto probablemente pertenece al Departamento: {int(prediccion)}")
                st.balloons()
            except Exception as e:
                st.error(f"Error al cargar el modelo: {e}")
        else:
            st.error(f"Error: El archivo `{nombre_archivo}` no se encuentra en la carpeta `/modelos`.")

# --- SECCIÓN 3: CONCLUSIONES ---
elif opcion == "Conclusiones":
    st.header("📌 Resumen de Hallazgos")
    st.markdown("""
    1. **Modelado:** El modelo **Hist-Gradient Boosting** ofreció un balance superior en la métrica F1-Score para predecir los departamentos más populares.
    2. **Fidelidad:** Un alto porcentaje de los productos en la canasta son re-compras (>59%), lo que sugiere una alta retención de clientes en categorías específicas.
    3. **Notebook:** Todos los pasos técnicos están documentados y pueden ser replicados en el enlace de **Google Colab** ubicado en el menú lateral.
    """)
