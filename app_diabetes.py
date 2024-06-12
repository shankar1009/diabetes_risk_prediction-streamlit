import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


import pandas as pd
import numpy as np







df=pd.read_csv('C:\\app\\diabetes_risk_prediction_dataset.csv')


df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1})
df['class'] = df['class'].replace({'Negative': 0, 'Positive': 1})

for column in df.columns.drop(['Age', 'Gender', 'class']):
    df[column] = df[column].replace({'No': 0, 'Yes': 1})

st.title("DIABETES CHECKUP")
x=df.drop(['class'],axis=1)
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


rf=RandomForestClassifier()
rf.fit(x_train, y_train)
def user_report():
    options1 = ['Male', 'Female']
    options = ['True', 'False']
    age = st.slider('Age',0,100,25)
    gender = st.radio("Select Gender", options1)
    polyuria = 1 if st.radio("Presence of excess urination", options) == 'True' else 0
    polydipsia = 1 if st.radio("Excessive thirst", options) == 'True' else 0
    suddenweight = 1 if st.radio("Sudden weight loss", options) == 'True' else 0
    weakness = 1 if st.radio("Weakness", options) == 'True' else 0
    polyphagia = 1 if st.radio("Excessive hunger", options) == 'True' else 0
    genitalthrush = 1 if st.radio("Presence of genital thrush", options) == 'True' else 0
    vision = 1 if st.radio("Blurring of vision", options) == 'True' else 0
    itch = 1 if st.radio("Presence of itching", options) == 'True' else 0
    irit = 1 if st.radio("Display of irritability", options) == 'True' else 0
    heal = 1 if st.radio("Delayed wound healing", options) == 'True' else 0
    move = 1 if st.radio("Partial loss of voluntary movement", options) == 'True' else 0
    muscle = 1 if st.radio("Muscle stiffness", options) == 'True' else 0
    hair = 1 if st.radio("Hair loss", options) == 'True' else 0
    obesity = 1 if st.radio("Presence of obesity", options) == 'True' else 0

    user_report = {
            'Age': age,
            'Gender': 0 if gender == 'Female' else 1,
            'Polyuria': int(polyuria),
            'Polydipsia': int(polydipsia),
            'sudden weight loss': int(suddenweight),
            'weakness': int(weakness),
            'Polyphagia': int(polyphagia),
            'Genital thrush': int(genitalthrush),
            'visual blurring': int(vision),
            'Itching': int(itch),
            'Irritability': int(irit),
            'delayed healing': int(heal),
            'partial paresis': int(move),
            'muscle stiffness': int(muscle),
            'Alopecia': int(hair),
            'Obesity': int(obesity)
    }

    report_data = pd.DataFrame(user_report, index=[0])
    return report_data
user_data=user_report()

st.write(user_data)
res=rf.predict(user_data)
st.subheader("Your Report:")
if res[0]==1:
    st.warning('You are not healthy')
else:
    st.success('You are  healthy')
    st.balloons()
