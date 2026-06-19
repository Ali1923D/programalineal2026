import streamlit as st
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

st.set_page_config(page_title="Optimización de CubeSat (MILP)", layout="wide")

st.title("🛰️ Optimizador de Misiones CubeSat mediante MILP")
st.markdown("""
Esta aplicación permite ajustar los parámetros y restricciones para la optimización del tiempo de operación 
de los componentes de un CubeSat utilizando Programación Lineal Entera Mixta (MILP).
""")

# CSS inyectado para obligar a las etiquetas (labels) a tener la misma altura y alinear los inputs
st.markdown("""
    <style>
    div[data-testid="stWidgetLabel"] {
        min-height: 48px;
        display: flex;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

st.write("---")

# Crear columnas para organizar la interfaz
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("🎯 Función Objetivo (Prioridades)")
    st.caption("Puntos otorgados por cada minuto de operación:")
    
    w1 = st.number_input("📸 Cámara (x1)", value=50)
    w2 = st.number_input("🔬 Espectrómetro (x2)", value=30)
    w3 = st.number_input("💻 Procesador (x3)", value=20)
    w4 = st.number_input("📡 Transmisor Banda X (x4)", value=40)
    w5 = st.number_input("📟 Transmisor UHF (x5)", value=10)
    w6 = st.number_input("❄️ Control Térmico (x6)", value=5)

with col2:
    st.header("⚙️ Límites de Restricciones")
    st.caption("Capacidades máximas y dependencias del sistema:")
    
    max_bateria = st.number_input("🔋 R1: Consumo Máximo Batería (mA*min)", value=15000)
    max_termico = st.number_input("🔥 R2: Presupuesto Térmico Máximo", value=2000)
    dep_proc = st.number_input("🔗 R3: Factor de dependencia Procesador (x3 ≥ f·x1)", value=0.50, step=0.05)
    max_descarga = st.number_input("⏳ R4: Ventana de Descarga UHF + Banda X (min)", value=35)
    max_bandax = st.number_input("⏱️ R5: Ciclo de trabajo Máximo Banda X (min)", value=20)
    
    # Input fantasma invisible para igualar la cantidad de filas en ambas columnas (6 vs 6)
    st.markdown('<div style="height: 77px;"></div>', unsafe_allow_html=True)

st.write("---")

# Resolver cuando el usuario presione el botón
if st.button("🚀 Ejecutar Optimización", type="primary", use_container_width=True):
    
    # Coeficientes (negativos para maximizar)
    c = [-w1, -w2, -w3, -w4, -w5, -w6]
    
    # Matriz de restricciones dinámica según los inputs
    A = [
        [250, 150, 100, 400, 80, 120],  # R1: Batería
        [20,   15,  30,  50, 10,   0],  # R2: Térmico
        [-dep_proc,  0,   1,   0,  0,   0],  # R3: Dependencia Procesador
        [0,     0,   0,   1,  1,   0],  # R4: Descarga
        [0,     0,   0,   1,  0,   0]   # R5: Banda X
    ]
    
    bu = [max_bateria, max_termico, np.inf, max_descarga, max_bandax]
    bl = [-np.inf, -np.inf, 0, -np.inf, -np.inf]
    
    constraints = LinearConstraint(A, bl, bu)
    bounds = Bounds([0]*6, [np.inf]*6)
    
    # Ejecución
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=[1]*6)
    
    st.header("📊 Resultados de la Optimización")
    
    if res.success:
        st.success(f"¡Solución Óptima Encontrada! Estado: {res.message}")
        st.metric(label="Valor Óptimo de la Función Objetivo (Z)", value=f"{-res.fun:.2f} puntos")
        
        componentes = [
            ("📸 Sensor de Imagen / Cámara (x1)", res.x[0]),
            ("🔬 Espectrómetro de Masas (x2)", res.x[1]),
            ("💻 Procesador de Bordo OBC (x3)", res.x[2]),
            ("📡 Transmisor de Radio Banda X (x4)", res.x[3]),
            ("📟 Transmisor de Radio UHF (x5)", res.x[4]),
            ("❄️ Control Térmico Activo (x6)", res.x[5])
        ]
        
        for nombre, valor in componentes:
            st.write(f"**{nombre}** = {int(round(valor))} minutos")
            
        cons_bat = sum(np.array(A[0]) * res.x)
        cons_term = sum(np.array(A[1]) * res.x)
        
        st.subheader("📈 Uso de Recursos Reales")
        st.progress(min(float(cons_bat/max_bateria), 1.0), text=f"Batería: {cons_bat:.0f} / {max_bateria} mA*min")
        st.progress(min(float(cons_term/max_termico), 1.0), text=f"Presupuesto Térmico: {cons_term:.0f} / {max_termico} unidades")
        
    else:
        st.error(f"El modelo no encontró una solución factible. Mensaje: {res.message}")
