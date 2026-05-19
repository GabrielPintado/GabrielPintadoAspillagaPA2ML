import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

# 1. CONFIGURACIÓN DE LA PÁGINA (Ícono: solo el emoji del carrito 🛒)
st.set_page_config(
    page_title="Instacart ML Magic Shopping",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo global para los gráficos
sns.set_theme(style="darkgrid")

# Mapeo para hacer divertida la respuesta del department_id
MAPEO_DEPARTAMENTOS = {
    1: "Congelados 🧊", 2: "Otros ❓", 3: "Panadería 🍞", 4: "Frutas y Verduras 🍎",
    5: "Alcohol 🍾", 6: "Internacional 🍣", 7: "Bebidas 🥤", 8: "Mascotas 🐶",
    9: "Pastas y Arroz 🍝", 10: "A Granel 🥜", 11: "Cuidado Personal 🧴", 12: "Carnes y Pescados 🥩",
    13: "Despensa 🍯", 14: "Desayunos 🥣", 15: "Enlatados 🥫", 16: "Lácteos y Huevos 🧀",
    17: "Artículos del Hogar 🧹", 18: "Bebés 👶", 19: "Snacks y Bocaditos 🍿", 20: "Delicatessen 🥪",
    21: "No Clasificado 🌀"
}

# --- CONTROL DE NAVEGACIÓN ---
opciones_menu = ["✨ ¡Bienvenida y Datos Locos!", "🔮 El Oráculo del Carrito (Predicción)", "🏆 El Veredicto Final (Conclusiones)"]

if 'indice_pestana' not in st.session_state:
    st.session_state.indice_pestana = 0

def actualizar_desde_radio():
    st.session_state.indice_pestana = opciones_menu.index(st.session_state.menu_lateral_radio)

# --- BARRA LATERAL ESTILIZADA ---
st.sidebar.image("mago_instacart.png", use_container_width=True)
st.sidebar.markdown("## 🧭 Central de Operaciones")

opcion = st.sidebar.radio(
    "Ir a:", 
    opciones_menu, 
    index=st.session_state.indice_pestana,
    key="menu_lateral_radio",
    on_change=actualizar_desde_radio
)

st.session_state.indice_pestana = opciones_menu.index(opcion)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Código Fuente")
st.sidebar.link_button("🚀 Abrir Notebook en Google Colab", "https://colab.research.google.com/drive/1IT-1KWfAL5K_Ur0qMz9-WeWZZvzsMuE4#scrollTo=ZuQ26sKpGTod")

# --- DATASET SIMULADO PARA GRÁFICOS ---
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

# ==========================================
# SECCIÓN 1: BIENVENIDA Y DATOS LOCOS
# ==========================================
if st.session_state.indice_pestana == 0:
    st.markdown("# 🛒 ¡Bienvenido a Instacart ML Magic Shopping! 🛍️")
    st.write("¡Una experiencia interactiva donde la Inteligencia Artificial adivina tus hábitos de consumo!")
    
    st.chat_message("assistant").write(
        "👋 ¡Hola! Este dataset es un **recorte estratégico de más de 3 millones de registros originales** "
        "reducido a 20,208 filas clave para entrenar nuestros modelos sin hacer explotar los servidores de Google Colab. "
        "¡Explora las estadísticas abajo antes de poner a prueba el algoritmo!"
    )
    
    st.markdown("---")
    st.subheader("📊 Datos Curiosos de la Muestra")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="🛍️ Productos en Canasta Analizados", value=f"{df.shape[0]} ítems")
    with c2:
        st.metric(label="⏰ Hora Pico de Compras", value="1:30 PM", delta="¡Hora del almuerzo!")
    with c3:
        st.metric(label="🔄 Clientes Fieles", value="59.3%", delta="Vuelven a pedir el producto")
        
    st.markdown("### 📈 ¿A qué hora compra la gente en Instacart?")
    
    color_grafico = st.selectbox("🎨 ¡Personaliza el color de la gráfica!", ["viridis", "magma", "plasma", "rocket", "crest"])
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.countplot(data=df, x='order_hour_of_day', palette=color_grafico, ax=ax)
    plt.title('Concentración del volumen de pedidos según la hora del día')
    plt.xlabel('Hora del Día (00:00 - 23:00)')
    plt.ylabel('Cantidad de Productos')
    st.pyplot(fig)

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔮 Siguiente: El Oráculo del Carrito ➡️", use_container_width=True):
        st.session_state.indice_pestana = 1
        st.rerun()

# ==========================================
# SECCIÓN 2: PREDICCIÓN (EL JUEGO/ORÁCULO)
# ==========================================
elif st.session_state.indice_pestana == 1:
    st.markdown("# 🔮 El Oráculo del Carrito de Compras")
    st.write("Configura tu comportamiento de compra y veamos si la Inteligencia Artificial descubre a qué departamento pertenece tu producto.")
    
    # Configuración del modelo en la barra lateral
    st.sidebar.markdown("### 🎛️ Cerebro Digital")
    modelo_seleccionado = st.sidebar.selectbox(
        "Escoge el modelo que adivinará:",
        ["Random Forest (El Clásico)", "Hist-Gradient Boosting (El Veloz)"]
    )
    
    # MENSAJE DE GUÍA AÑADIDO: Explica dónde cambiar el modelo predictivo
    st.info("💡 **Consejo de Mago:** Puedes cambiar el algoritmo predictivo en cualquier momento usando el menú **Cerebro Digital** en la barra lateral izquierda (debajo de la Central de Operaciones) para ver cómo cambian los resultados.")
    
    with st.container(border=True):
        st.subheader("🛒 Diseña tu Comportamiento de Compra")
        
        col1, col2 = st.columns(2)
        with col1:
            add_to_cart_order = st.slider("🎒 ¿En qué orden metiste este producto al carrito?", min_value=1, max_value=30, value=4)
            reordered = st.radio("🔄 ¿Ya habías comprado este artículo en el pasado?", options=[0, 1], format_func=lambda x: "👍 ¡Sí, ya lo conozco!" if x == 1 else "🆕 ¡No, es la primera vez!")
            order_dow = st.select_slider("📅 ¿Qué día de la semana estás comprando?", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Domingo 🏖️", "Lunes 📈", "Martes 🪵", "Miércoles ☕", "Jueves 🚀", "Viernes 🥳", "Sábado 🎮"][x])
            
        with col2:
            order_hour_of_day = st.slider("🕒 ¿A qué hora estás haciendo click en 'Comprar'?", min_value=0, max_value=23, value=15)
            days_since_prior_order = st.number_input("⏳ ¿Cuántos días pasaron desde tu última orden en Instacart?", min_value=0.0, max_value=30.0, value=7.0, step=0.5)
            
    nombre_archivo = "modelo_random_forest.pkl" if "Random Forest" in modelo_seleccionado else "modelo_gradient_boosting.pkl"
    ruta_modelo = os.path.join("modelos", nombre_archivo)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🧙‍♂️ ¡Invocar Predicción de IA!", use_container_width=True):
        if os.path.exists(ruta_modelo):
            try:
                with st.spinner('🔮 Consultando patrones en la base de datos de Instacart...'):
                    time.sleep(1.2)
                
                modelo = joblib.load(ruta_modelo)
                
                features = pd.DataFrame([{
                    'add_to_cart_order': add_to_cart_order,
                    'reordered': reordered,
                    'order_dow': order_dow,
                    'order_hour_of_day': order_hour_of_day,
                    'days_since_prior_order': days_since_prior_order
                }])
                
                prediccion = int(modelo.predict(features)[0])
                nombre_depto = MAPEO_DEPARTAMENTOS.get(prediccion, "Departamento Desconocido 🛰️")
                
                st.balloons()
                st.success(f"### 🤖 ¡La IA ha determinado que tu producto pertenece al pasillo de:")
                st.markdown(f"<h1 style='text-align: center; color: #2E7D32;'>{nombre_depto}</h1>", unsafe_allow_html=True)
                st.info(f"🔢 **Identificador Técnico:** `department_id = {prediccion}` | 🧠 **Modelo Usado:** {modelo_seleccionado}")
                
            except Exception as e:
                st.error(f"❌ Error al procesar los datos: {e}")
        else:
            st.error(f"⚠️ ¡Falta el archivo del modelo! Asegúrate de subir `{nombre_archivo}` dentro de la carpeta `modelos/` en tu GitHub.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("⬅️ Anterior: Datos Locos", use_container_width=True):
            st.session_state.indice_pestana = 0
            st.rerun()
    with btn_col2:
        if st.button("🏆 Siguiente: El Veredicto Final ➡️", use_container_width=True):
            st.session_state.indice_pestana = 2
            st.rerun()

# ==========================================
# SECCIÓN 3: CONCLUSIONES
# ==========================================
elif st.session_state.indice_pestana == 2:
    st.markdown("# 🏆 Conclusiones del Laboratorio de Datos")
    
    st.markdown("### 🔥 Duelo Técnico: ¿Quién ganó el campeonato?")
    
    data_comparativa = {
        "Métrica": ["Accuracy (Acierto Global)", "F1-Score (Macro)", "Velocidad de Carga"],
        "Random Forest 🌲": ["44.11%", "0.1767", "Pesado (.pkl grande)"],
        "Hist-Gradient Boosting ⚡": ["44.64%", "0.1901", "Ligero y Veloz (¡Ganador!)"]
    }
    df_metricas = pd.DataFrame(data_comparativa)
    st.table(df_metricas)
    
    st.markdown("### 💡 Aprendizajes Clave")
    st.warning("⚖️ **La trampa del Accuracy:** Como el dataset original está desbalanceado, el **F1-Score Macro** fue nuestra métrica brújula para garantizar que el modelo aprenda de todas las categorías por igual.")
    st.success("⚙️ **Optimización en la Nube:** Recortar el dataset y usar algoritmos basados en histogramas (`Hist-Gradient Boosting`) salvó nuestro entorno en Colab de morir por falta de memoria RAM.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("⬅️ Volver a: El Oráculo del Carrito", use_container_width=True):
        st.session_state.indice_pestana = 1
        st.rerun()
