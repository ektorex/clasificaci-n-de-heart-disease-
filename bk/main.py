#! /usr/bin/env -S streamlit run

import joblib
import matplotlib as plt
import numpy as np 
import pandas as pd
import streamlit as st 

@st.cache_resource
def cargarModelo():
    modelo = joblib.load('modelo.pkl')
    return modelo

modelo_cargado = cargarModelo()

def main():
    # st.title("title")
    #
    # st.header("head")
    #
    # st.subheader("sh")
    #
    # st.caption("sub")
    #
    #
    # st.divider()
    #
    # data = pd.DataFrame(
    #     np.random.randn(20,4),
    #     columns=['A','B','C','D']
    # )
    #
    # st.area_chart(data)
    #
    #
    # st.bar_chart(data)
    # st.line_chart(data)
    # st.scatter_chart(data)
    #
    # map_data = pd.DataFrame(
    #     np.random.randn(100,2) / [50,50] + [37.76, -122.4],
    #     columns=['lat','lon']
    # )
    #
    # st.map(map_data)

    # st.divider()

    values = {
        "age": None,
        "sex": None,
        "cp": None,
        "trestbps": None,
        "chol": None,
        "fbs": None,
        "restecg": None,
        "thalch": None,
        "exang": None,
        "oldpeak": None,
        "slope": None,
        "ca": None,
        "thal": None,
    }

    Materia = "Data Science"

    Asesor=  "Dra. Olanda Prieto Ordaz"

    Nombre= "Héctor Rodríguez Loya"

    Matricul=  "363325"

    st.title(Materia)

    with st.form(key="form"):
        values["age"] = st.number_input("Edad")
        values["sex"] = st.selectbox("Genero", ["Male","Female"])
        values["cp"] = st.selectbox("cp", ['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'])
        values["trestbps"] = st.number_input("trestbps")
        values["chol"] = st.number_input("chol")
        values["fbs"] = st.selectbox("fbs", ["True","False"])
        values["restecg"] = st.selectbox("restecg", ['lv hypertrophy', 'normal', 'st-t abnormality'])
        values["thalch"] = st.number_input("thalch") 
        values["exang"] = st.selectbox("exang", ["True","False"])
        values["oldpeak"] = st.number_input("oldpeak")
        values["slope"] = st.selectbox("slope", ['downsloping', 'flat', 'upsloping']) 
        values["ca"] = st.number_input("ca")
        values["thal"] = st.selectbox("thal",['fixed defect', 'normal', 'reversable defect'])

        summit = st.form_submit_button(label="Consultar")
        if summit:
            if not all(values.values()):
                st.warning("llena todo")
            else:
                imput = pd.DataFrame([values])
                prediccion = modelo_cargado.predict(imput)
                st.balloons()
                st.write(prediccion)

if __name__ == "__main__":
    main()
