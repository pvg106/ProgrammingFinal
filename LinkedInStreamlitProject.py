#importing packages
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image

#heading and image for the streamlit application
pic = Image.open('Linkedin_Image.png')
st.markdown("# LinkedIn User Prediction Application")
st.markdown("#### Created by: Philip Gasparovic")
st.markdown("##### Please use the dropdown boxes and slider below to start your prediction.")

#reading in the csv file for the logical regresson model
s = pd.read_csv("social_media_usage.csv")

#feature engineering to make imported csv data usable
ss = pd.DataFrame({
    "sm_li": np.where(s["web1h"] == 1, 1, 0),
    "income": np.where(s["income"] <=9, s["income"], np.nan),
    "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"]==2, 1, 0),
    "age": np.where(s["age"] < 98, s["age"], np.nan)
})

#removing all na values
ss = ss.dropna()

#establishing target and predictor variables
y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]

#dividing the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   random_state=123)

#initiating the balanced LR model                                    
lr = LogisticRegression(class_weight = "balanced")

#fitting the model
lr.fit(x_train, y_train)

#using the test data to make predictions on the model
y_pred = lr.predict(x_test)

#income selection dropdown and writing value
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

#education selection dropdown and writing value
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

#parent selection dropdown and writing value
parent = st.selectbox("Are you a parent",
             options = ["Yes",
                        "No"])
st.write(f"###### You chose: {parent}")

#married selection dropdown and writing value
married = st.selectbox("Are you married",
             options = ["Yes",
                        "No"])
st.write(f"###### You chose: {married}")

#gender selection dropdown and writing value
female = st.selectbox("Gender",
             options = ["Female",
                        "Male"])
st.write(f"###### You chose: {female}")

#age slider bar and writing value
age = st.slider(label = "Please use the slider to select your age (1-97 years old)",
                min_value = 1,
                max_value = 97,
                value = 21)
st.write(f"###### You Chose: {age}")

#changing income variable into a usable numeric value
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

#changing educ variable into a usable numeric value
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

#changing parent variable into a usable numeric value
if parent == "Yes":
    parent = 1
else:
    parent = 0

#changing married variable into a usable numeric value
if married == "Yes":
    married = 1
else:
    married = 0

#changing gender variable into a usable numeric value
if female == "Female":
    female = 1
else:
    female = 0

#building in a button feature to submit the selected values for prediciton and to return the prediction
if st.button("Predict"):
    person = (income, education, parent, married, female, age)
    predicted_class = lr.predict([person])
    probs = lr.predict_proba([person])

#writing the prediction results if the user is a LinkedIn user or not
if predicted_class[0] == 1:
    st.write(f"### This person is a LinkedIn User")
else:
    st.write(f"### This person is not a LinkedIn User")

#shows the probability that this person is a LinkedIn user
st.write(f"#### Probability that this person is a LinkedIn user: {probs[0][1]}")