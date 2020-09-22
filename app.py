import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

@st.cache
def get_data():
  return pd.read_csv('data.csv')

def train_model():
  data = get_data()
  x = data.drop("MEDV",axis=1)
  y = data["MEDV"]
  rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
  rf_regressor.fit(x, y)
  return rf_regressor

data = get_data()

model = train_model()

st.title('Data App - Predicting House Values')

st.markdown('')

st.subheader('Selecting the small columns set')

default_cols = ['RM', 'PTRATIO', 'LSTAT', 'MEDV']

cols = st.multiselect("Columns", data.columns.tolist(), default=default_cols)

st.dataframe(data[cols].head(10))

st.subheader('Distplot - Houses per Value')

values = st.slider('Value Range', float(data.MEDV.min()), 150., (10.0, 100.0))

dt = data[data['MEDV'].between(left=values[0], right=values[1])]

f = px.histogram(dt, x='MEDV', nbins=100, title='Distplot of Values')
f.update_xaxes(title='MEDV')
f.update_yaxes(title='Total Houses')
st.plotly_chart(f)

st.sidebar.subheader('Choose the attributes of the house for prediction')

crim = st.sidebar.number_input('Criminal Tax', value=data.CRIM.mean())
indus = st.sidebar.number_input('Business Area Proportion', value=data.INDUS.mean())
chas = st.sidebar.selectbox('Does it border the river?', ('Yes', 'No'))

chas = 1 if chas == 'Yes' else 0

nox = st.sidebar.number_input('Nitrix Concentration', value=data.NOX.mean())
rm = st.sidebar.number_input('Room Numbers', value=1)
ptratio = st.sidebar.number_input('Teachers for Students Index', value=data.PTRATIO.mean())
b = st.sidebar.number_input('Afro Americans Proportion', value=data.B.mean())
lstat = st.sidebar.number_input('Proportion of Low Status Population', value=data.LSTAT.mean())

btn_predict = st.sidebar.button('Predict')

if btn_predict:
  result = model.predict([[crim, indus, chas, nox, rm, ptratio, b, lstat]])
  st.subheader('The value for the house predicted is:')
  result = 'U$ ' + str(round(result[0] * 10, 2))
  st.write(result)

