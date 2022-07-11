import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(layout="wide")

with open('style.css') as f:
  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

df = pd.read_csv('Data/car_price_data.csv')
df = df.dropna()

def preprocess(column: str):
    return sorted([i.title().replace('_',' ') for i in df[column].unique()])

with st.form("my_form"):
    a1, a2, a3 = st.columns((5,5,5))
    with a1:
        make = st.selectbox('Choose a BRAND', preprocess('Make'))
        model = st.selectbox('Choose a MODEL', preprocess('Model'))
        engine_fuel_type = st.selectbox('Choose an ENGINE FUEL TYPE', preprocess('Engine Fuel Type'))
        engine_cylinders = st.number_input('Insert NUMBER of ENGINE CYLINDER', min_value=0)
        transmission_type = st.selectbox('Choose a TRANSMISSION TYPE', preprocess('Transmission Type'))
    with a2:
        driven_wheels = st.selectbox('Choose a DRIVEN WHEELS', preprocess('Driven_Wheels'))
        number_of_doors = st.number_input('Insert NUMBER of DOORS', min_value=2)
        market_category = st.multiselect('Choose a MARKET CATEGORY', sorted(set([j for sub in [i.split(',') for i in df['Market Category'].unique()] for j in sub])), default='Luxury')
        vehicle_size = st.selectbox('Choose a VEHICLE SIZE', preprocess('Vehicle Size'))
        vehicle_style = st.selectbox('Choose a VEHICLE STYLE', preprocess('Vehicle Style'))
    with a3:
        year = st.selectbox('Choose a PRODUCTION YEAR', [i for i in range(1920,2017)])
        engine_hp = st.number_input('Insert ENGINE HP', min_value=0)
        highway_mpg = st.number_input('Insert HIGHWAY MPG', min_value=0)
        city_mpg = st.number_input('Insert CITY MPG', min_value=0)
        popularity = st.number_input('Insert POPULARITY', min_value=0)

    new_data = {'make': make.lower(),
                'model': model.lower().replace(' ','_'),
                'year': 2018-int(year),
                'engine_fuel_type': engine_fuel_type.lower().replace(' ','_'),
                'engine_hp': engine_hp,
                'engine_cylinders': str(engine_cylinders),
                'transmission_type': transmission_type.lower().replace(' ','_'),
                'driven_wheels': driven_wheels.lower().replace(' ','_'),
                'number_of_doors': str(number_of_doors),
                'market_category': ",".join(market_category).lower().replace(' ','_'),
                'vehicle_size': vehicle_size.lower().replace(' ','_'),
                'vehicle_style': vehicle_style.lower().replace(' ','_'),
                'highway_mpg': highway_mpg,
                'city_mpg': city_mpg,
                'popularity': popularity
    }

    xgb_reg = XGBRegressor()
    xgb_reg.load_model('Models/xgb_reg.json')

    df_new = pd.DataFrame([new_data])
    df_new[df_new.select_dtypes(include=['int', 'float']).columns] = np.log1p(df_new.select_dtypes(include=['int', 'float'])[0:1])
    df_dummy = pd.get_dummies(df_new).T.reset_index().rename(columns={'index':'col_name', 0:'value'})
    df_dummy1 = pd.DataFrame({'col_name':xgb_reg.feature_names_in_})
    df_dummy2 = df_dummy1.merge(df_dummy, on='col_name', how='outer').fillna(0)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        # st.write(new_data)
        # st.dataframe(pd.DataFrame([new_data]))
        st.write("The Predicted Price of The Car is USD {:.2f}".format(np.expm1(xgb_reg.predict(df_dummy2.set_index('col_name').T.values))[0]))