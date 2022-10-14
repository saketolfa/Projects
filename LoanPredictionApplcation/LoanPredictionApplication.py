#import libraries
import math
import pandas as pd
import streamlit as st
from PIL import Image
import pickle

# Build 2 pages  Home and Prediction
app_mode = st.sidebar.selectbox('Select page' , ['Home', 'Prediction'])

# Home page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center; color: grey;'>Welcome to  loan prediction </h1>", unsafe_allow_html=True)
    # Home image
    image = Image.open('./loan_image.jpg')
    width, heigth = image.size
    image = image.resize((width, heigth // 2))
    st.image(image)
    # dataset
    st.markdown("<h4> Dataset: </h4>",
                unsafe_allow_html=True)

    # read a csv file
    df = pd.read_csv('home loan/train.csv')
    st.write(df.head(10))
    # dimension of our dataset
    st.write('Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.' )
    st.write(" For  " + str(df.shape[0]) + " Costumers, " + str(math.ceil((df['Loan_Status'].value_counts(normalize = True)*100)[0])) + "%  are eligible for the loan amount and "
             + str(math.ceil((df['Loan_Status'].value_counts(normalize = True) *100)[1] ))+ "%  are not.")



# Prediction page
elif app_mode=='Prediction':
    st.markdown("<h1 style='text-align: center; color: grey;'>Let's doing some prediction</h1>", unsafe_allow_html=True)
    # Prediction image
    image = Image.open('./prediction_image.jpg')
    width, heigth = image.size
    image = image.resize((width + 300, heigth //2 ))
    st.image(image)
    st.write('We predict whether you loan should be approved or not. ')
    st.write('You need to fill all necessary information in order  to get a reply to your loan request !')
    st.sidebar.header('Informations about the client:')

    # input data's client
    gender_dict = {'Male':0 , 'Female':1}
    yn_dict = {'Yes': 1 , 'No': 0}
    credit_history_dict = {'Yes': 1.0 , 'No': 0.0}
    property_area_dict = { 'Rural':0,'Urban':1,'Semiurban':2}
    dependents_dict ={'0':0 , '1':1 ,'2':2 ,'3+':3}

    gender = st.sidebar.radio('Gender: ' , tuple(gender_dict.keys()))
    maried = st.sidebar.radio('Maried: ', tuple(yn_dict.keys()))
    dependents = st.sidebar.radio('Dependents:' ,  ['0','1','2','3+'] )
    education = st.sidebar.radio('Education: ', tuple(yn_dict.keys()))
    self_employed = st.sidebar.radio('Self_employed: ', tuple(yn_dict.keys()))
    applicantincome = st.sidebar.slider('ApplicantIncome', 0, 10000 , 5000)
    coapplicantincome =st.sidebar.slider('CoApplicantIncome' , 0 , 10000 , 5000)
    loan_amount = st.sidebar.slider('loan_amount' ,9.0 , 700.0 , 300.0 )
    loan_amount_term = st.sidebar.selectbox('Loan_Amount_Term' , (12.0,36.0,60.0,84.0,120.0,180.0,240.0,300.0,360.0))
    property_area = st.sidebar.radio('Property area:' , tuple(property_area_dict))
    credit_history = st.sidebar.radio(' Credit_history' , tuple(credit_history_dict.keys()))


    data_client_dicts = {
            'Gender': gender_dict[gender],
            'Married':yn_dict[maried],
            'Dependents':dependents_dict[dependents],
            'Education':yn_dict[education],
            'self Employed': yn_dict[self_employed],
            'ApplicantIncome': applicantincome,
            'CoapplicantIncome': coapplicantincome,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Property_Area': property_area_dict[property_area],
            'Credit_History': credit_history_dict[credit_history],

     }
    # build a dataFrame from data client
    df =pd.DataFrame(data_client_dicts , index=[0])

    # dispaly the imput data
    st.markdown("<h4> Your input data </h4>",
                unsafe_allow_html=True)
    st.write (df)


    # pickle the model
    model = pickle.load(open('loan_predict.pkl', 'rb'))
    prediction = model.predict(df)
    probability = model.predict_proba(df)


    # Prediction Button
    if st.button('Prediction'):
        if prediction[0] ==0:
            st.error('you have a  '+ str(math.ceil(probability[0][0]*100)) +  ' % probability to not get  the loan from Bank')
        elif prediction[0] == 1:
            st.success(' Congratulations!! you have a' + str(math.ceil(probability[0][1]*100)) + ' % probability to get the loan from Bank ')

