import streamlit as st
import pickle

st.title('Diabetes prediction using simple Decision-Tree-Classification')
#Pregnancies
Pregnancies = st.number_input(
    label='Pregnancies:'
 )
# Glucose,
Glucose = st.number_input(
    label='Glucose:'
 )
# BloodPressure,
BloodPressure = st.number_input(
    label='BloodPressure:'
 )
# SkinThickness,
SkinThickness = st.number_input(
    label='SkinThickness:'
 )
# Insulin,
Insulin = st.number_input(
    label='Insulin:'
 )
# BMI,
BMI = st.number_input(
    label='BMI:'
 )
# DiabetesPedigreeFunction,
DiabetesPedigreeFunction = st.number_input(
    label='DiabetesPedigreeFunction:'
 )
# Age,
Age = st.number_input(
    label='Age:'
 )
pt= st.button('submit')
if pt== True:
    with open('classifier.pkl', 'rb') as file:
        model = pickle.load(file)
    predict = model.predict([[Pregnancies,Glucose,BloodPressure ,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    st.write('Output label:')
    st.write(predict)
