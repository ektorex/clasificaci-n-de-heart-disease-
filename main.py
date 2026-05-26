#! /usr/bin/env -S streamlit run

import joblib
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Salud Cardiovascular",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def cargarModelo():
    try:
        modelo = joblib.load('modelo.pkl')
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo_cargado = cargarModelo()

def main():
    # Información lateral (Sidebar)
    st.sidebar.image("https://www.titulosydiplomas.com/cdn/shop/collections/universidad-autonoma-de-chihuahua.png?v=1739470084", width=300) # Ejemplo de logo
    st.sidebar.title("Información del Proyecto")
    
    materia = "Data Science"
    asesor = "Dra. Olanda Prieto Ordaz"
    nombre = "Héctor Rodríguez Loya"
    matricula = "363325"

    st.sidebar.markdown(f"""
    **Materia:** {materia}  
    **Asesor:** {asesor}  
    **Nombre:** {nombre}  
    **Matrícula:** {matricula}
    """)

    st.sidebar.divider()
    st.sidebar.info("""
    Esta aplicación utiliza un modelo de Machine Learning para predecir la presencia de enfermedades cardíacas 
    basándose en indicadores clínicos.
    """)

    # Título principal
    st.title("Predicción de Salud Cardiovascular")
    st.markdown("---")

    # Explicación de los datos en un expansor
    with st.expander("Información sobre las variables de entrada"):
        st.write("""
        Los datos se basan en el conjunto de datos de Cleveland. Aquí se detallan los parámetros:
        - **Edad:** Edad del paciente en años.
        - **Género:** Sexo biológico del paciente.
        - **CP (Chest Pain):** Tipo de dolor de pecho (Angina típica, asintomática, no anginal, angina atípica).
        - **Trestbps:** Presión arterial en reposo (en mm Hg al ingreso al hospital).
        - **Chol:** Colesterol sérico en mg/dl.
        - **FBS:** Glucemia en ayunas > 120 mg/dl (Verdadero/Falso).
        - **Restecg:** Resultados electrocardiográficos en reposo.
        - **Thalach:** Frecuencia cardíaca máxima alcanzada.
        - **Exang:** Angina inducida por el ejercicio (Sí/No).
        - **Oldpeak:** Depresión del segmento ST inducida por el ejercicio en relación con el reposo.
        - **Slope:** Pendiente del segmento ST en el ejercicio máximo.
        - **CA:** Número de vasos principales (0-3) coloreados por fluoroscopia.
        - **Thal:** Estado del corazón (Normal, defecto fijo, defecto reversible).
        """)

    # Formulario de entrada
    with st.form(key="form_prediccion"):
        st.subheader("Ingrese los Datos Clínicos")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Edad", min_value=1, max_value=120, value=50, help="Edad en años")
            sex = st.selectbox("Género", ["Male", "Female"], help="1 = masculino; 0 = femenino")
            cp = st.selectbox("Tipo de Dolor (CP)", 
                              ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'],
                              help="Tipo de dolor de pecho reportado")
            trestbps = st.number_input("Presión Arterial (trestbps)", min_value=50, max_value=250, value=120, 
                                      help="Presión arterial en reposo (mm Hg)")

        with col2:
            chol = st.number_input("Colesterol (chol)", min_value=100, max_value=600, value=200, 
                                   help="Colesterol sérico en mg/dl")
            fbs = st.selectbox("Azúcar en Ayunas > 120 (fbs)", ["False", "True"], 
                               help="¿El azúcar en ayunas es mayor a 120 mg/dl?")
            restecg = st.selectbox("Electrocardiograma (restecg)", 
                                   ['normal', 'st-t abnormality', 'lv hypertrophy'],
                                   help="Resultados electrocardiográficos en reposo")
            thalch = st.number_input("Frecuencia Cardíaca Máx (thalch)", min_value=50, max_value=220, value=150,
                                     help="Máxima frecuencia cardíaca alcanzada")

        with col3:
            exang = st.selectbox("Angina por Ejercicio (exang)", ["False", "True"], 
                                 help="¿Angina inducida por el ejercicio?")
            oldpeak = st.number_input("Depresión ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                      help="Depresión del ST inducida por ejercicio relativo al reposo")
            slope = st.selectbox("Pendiente ST (slope)", ['upsloping', 'flat', 'downsloping'],
                                 help="Pendiente del segmento ST en ejercicio máximo")
            ca = st.number_input("Vasos Coloreados (ca)", min_value=0, max_value=3, value=0,
                                 help="Número de vasos principales (0-3) coloreados por fluoroscopia")
            thal = st.selectbox("Thal", ['normal', 'fixed defect', 'reversable defect'],
                                help="3 = normal; 6 = defecto fijo; 7 = defecto reversible")

        st.markdown("---")
        submit = st.form_submit_button(label="Realizar Predicción")

    if submit:
        # Preparar los datos para el modelo
        values = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalch": thalch,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        
        input_df = pd.DataFrame([values])
        
        with st.spinner('Procesando predicción...'):
            if modelo_cargado is not None:
                prediccion = modelo_cargado.predict(input_df)
                
                st.subheader("Resultado de la Evaluación")
                
                # El modelo predice de 0 a 4:
                # 0: No heart disease
                # 1-4: Etapas de la enfermedad
                res = int(prediccion[0])
                
                if res == 0:
                    st.success("✅ **Resultado: No se detectó enfermedad cardíaca (Estado 0).**")
                    st.balloons()
                else:
                    st.error(f"⚠️ **Resultado: Presencia de enfermedad cardíaca detectada (Etapa {res}).**")
                    
                    if res == 1:
                        st.warning("Etapa 1: Etapa inicial detectada. Se recomienda monitoreo médico.")
                    elif res == 2:
                        st.warning("Etapa 2: Etapa intermedia detectada. Requiere atención médica.")
                    elif res == 3:
                        st.warning("Etapa 3: Etapa avanzada detectada. Atención médica prioritaria necesaria.")
                    elif res == 4:
                        st.error("Etapa 4: Etapa crítica detectada. Requiere intervención médica inmediata.")
                    
                    st.info("Por favor, consulte a un especialista para una evaluación clínica completa.")
                
                # Mostrar los datos ingresados para referencia
                with st.expander("Ver datos de entrada"):
                    st.dataframe(input_df)
            else:
                st.error("El modelo no está disponible.")

if __name__ == "__main__":
    main()
