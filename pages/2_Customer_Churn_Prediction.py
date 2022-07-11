import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

df = pd.read_csv('Data/telco_customer_churn.csv')
df = df.dropna()
df['SeniorCitizen'] = df['SeniorCitizen'].replace({0:'No',1:'Yes'})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

def preprocess(column: str):
    return sorted([i.title().replace('_',' ') for i in df[column].unique()])

with st.form("my_form"):
    a1, a2, a3 = st.columns((5,5,5))
    with a1:
        gender = st.selectbox('Choose a Gender', preprocess('gender'))
        seniorcitizen = st.selectbox('Choose a senior citizen', preprocess('seniorcitizen'))
        partner = st.selectbox('Choose an partner', preprocess('partner'))
        dependents = st.selectbox('Choose a dependents', preprocess('dependents'))
        phoneservice = st.selectbox('Choose a phoneservice', preprocess('phoneservice'))
        multiplelines = st.selectbox('Choose a multiplelines', preprocess('multiplelines'))
        internetservice = st.selectbox('Choose a internet service', preprocess('internetservice'))
    with a2:
        onlinesecurity = st.selectbox('Choose a online security', preprocess('onlinesecurity'))
        onlinebackup = st.selectbox('Choose a online backup', preprocess('onlinebackup'))
        deviceprotection = st.selectbox('Choose a device protection', preprocess('deviceprotection'))
        techsupport = st.selectbox('Choose a tech support', preprocess('techsupport'))
        streamingtv = st.selectbox('Choose a streaming tv', preprocess('streamingtv'))
        streamingmovies = st.selectbox('Choose a streaming movies', preprocess('streamingmovies'))
    with a3:
        contract = st.selectbox('Choose a contract', preprocess('contract'))
        paperlessbilling = st.selectbox('Choose a paperless billing', preprocess('paperlessbilling'))
        paymentmethod = st.selectbox('Choose a payment method', preprocess('paymentmethod'))
        tenure = st.number_input('Insert tenure', min_value=0)
        monthlycharges = st.number_input('Insert monthly charges')
        totalcharges = st.number_input('Insert total charges')

    new_data = {'gender': gender.lower().replace(' ','_'),
                'seniorcitizen': seniorcitizen.lower().replace(' ','_'),
                'partner': partner.lower().replace(' ','_'),
                'dependents': dependents.lower().replace(' ','_'),
                'phoneservice': phoneservice.lower().replace(' ','_'),
                'multiplelines': multiplelines.lower().replace(' ','_'),
                'internetservice': internetservice.lower().replace(' ','_'),
                'onlinesecurity': onlinesecurity.lower().replace(' ','_'),
                'onlinebackup': onlinebackup.lower().replace(' ','_'),
                'deviceprotection': deviceprotection.lower().replace(' ','_'),
                'techsupport': techsupport.lower().replace(' ','_'),
                'streamingtv': streamingtv.lower().replace(' ','_'),
                'streamingmovies': streamingmovies.lower().replace(' ','_'),
                'contract': contract.lower().replace(' ','_'),
                'paperlessbilling': paperlessbilling.lower().replace(' ','_'),
                'paymentmethod': paymentmethod.lower().replace(' ','_'),
                'tenure': tenure,                
                'monthlycharges': monthlycharges,                
                'totalcharges': totalcharges   
    }

    xgb_clf = XGBClassifier()
    xgb_clf.load_model('Models/xgb_clf.json')

    df_new = pd.DataFrame([new_data])

    def transform(df, numerical, binary):
        for i in numerical:
            df[i] = np.log1p(df[i].values)

        for i in binary:
            df[i] = (df[i]=='yes').astype(int)

        transformed = pd.get_dummies(df)
        return transformed

    numerical = list(df.select_dtypes(include=['int','float']).columns)
    binary = list(i for i in df.columns if df[i].nunique()==2)

    df_new_transformed = transform(df_new, numerical, binary[:-1])
    df_dummy = pd.get_dummies(df_new_transformed).T.reset_index().rename(columns={'index':'col_name', 0:'value'})
    df_dummy1 = pd.DataFrame({'col_name':xgb_clf.feature_names_in_})
    df_dummy2 = df_dummy1.merge(df_dummy, on='col_name', how='outer').fillna(0)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        # st.dataframe(df_new)
        prediction = xgb_clf.predict(df_dummy2.set_index('col_name').T.values)[0]
        st.write("The Customer will {}".format('Not Churn' if prediction==0 else 'Churn'))
