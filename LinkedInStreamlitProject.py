import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.markdown("# LinkedIn User Prediction Application")
st.markdown("#### Created by: Philip Gasparovic")
st.markdown("##### Please use the dropdown boxes and slider below to start your prediction.")

s = pd.read_csv("social_media_usage.csv")

ss = pd.DataFrame({
    "sm_li": np.where(s["web1h"] == 1, 1, 0),
    "income": np.where(s["income"] <=9, s["income"], np.nan),
    "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"]==2, 1, 0),
    "age": np.where(s["age"] < 98, s["age"], np.nan)
})

ss = ss.dropna()

y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   random_state=123)
                                    
lr = LogisticRegression(class_weight = "balanced")

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

income = st.selectbox("Income (in dollars)", 
             options =  ["Less than 10,000",
                        "10,000 to under 20,000",
                        "20,000 to under 30,000",
                        "30,000 to under 40,000",
                        "40,000 to under 50,000",
                        "50,000 to under 75,000",
                        "75,000 to under 100,000",
                        "100,000 to under 150,000",
                        "150,000 or more"])
st.write(f"###### You chose: {income}")

educ = st.selectbox("Education level", 
             options = ["Less than High School",
                        "High School Incomplete",
                        "High School Diploma",
                        "Some College, No Degree",
                        "Two-Year Associates Degree",
                        "Four-Year Bachelor's Degree",
                        "Some Postgraduate or Professional Schooling",
                        "Postgraduate or Professional Degree"])
st.write(f"###### You chose: {educ}")

parent = st.selectbox("Are you a parent",
             options = ["Yes",
                        "No"])
st.write(f"###### You chose: {parent}")

married = st.selectbox("Are you married",
             options = ["Yes",
                        "No"])
st.write(f"###### You chose: {married}")

female = st.selectbox("Gender",
             options = ["Female",
                        "Male"])
st.write(f"###### You chose: {female}")

age = st.slider(label = "Please use the slider to select your age (1-97 years old)",
                min_value = 1,
                max_value = 97,
                value = 21)
st.write(f"###### You Chose: {age}")


if income == "Less than 10,000":
    income = 1
elif income == "10,000 to under 20,000":
     income = 2
elif income == "20,000 to under 30,000":
     income = 3
elif income == "30,000 to under 40,000":
     income = 4
elif income == "40,000 to under 50,000":
     income = 5
elif income == "50,000 to under 75,000":
     income = 6
elif income == "75,000 to under 100,000":
     income = 7
elif income == "100,000 to under 150,000":
     income = 8
else:
     income = 9


if educ == "Less than High School":
    education = 1
elif educ == "High School Incomplete":
     education = 2
elif educ == "High School Diploma":
     education = 3
elif educ == "Some College, No Degree":
     education = 4
elif educ == "Two-Year Associates Degree":
     education = 5
elif educ =="Four-Year Bachelor's Degree":
     education = 6
elif educ == "Some Postgraduate or Professional Schooling":
     education = 7
else:
     education = 8


if parent == "Yes":
    parent = 1
else:
    parent = 0


if married == "Yes":
    married = 1
else:
    married = 0


if female == "Female":
    female = 1
else:
    female = 0


person = (income, education, parent, married, female, age)

predicted_class = lr.predict([person])
probs = lr.predict_proba([person])

if predicted_class[0] == 1:
    st.write(f"### This person is a LinkedIn User")
else:
    st.write(f"### This person is not a LinkedIn User")

st.write(f"#### Probability that this person is a LinkedIn user: {probs[0][1]}")