import streamlit as st
import pickle
import numpy as np

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

st.title("💻 Laptop Price Predictor")
st.write("Fill in the laptop specifics below to get an estimated price (INR).")

# Split UI properly
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', sorted(df['Company'].unique()))
    type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
    ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
    os = st.selectbox('Operating System', sorted(df['os'].unique()))
    cpu = st.selectbox('CPU Brand', sorted(df['Cpu brand'].unique()))
    gpu = st.selectbox('GPU Brand', sorted(df['Gpu brand'].unique()))

with col2:
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Display', ['No', 'Yes'])
    screen_size = st.number_input('Screen Size (Inches)', min_value=10.0, max_value=18.0, value=15.6, step=0.1)
    resolution = st.selectbox('Screen Resolution', 
                              ['1920x1080', '1366x768', '1600x900', '3840x2160', 
                               '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# Storage
st.subheader("Storage Options")
col3, col4 = st.columns(2)
with col3:
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2000])
with col4:
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

if st.button('Predict Price', use_container_width=True, type='primary'):
    # Pre-processing user inputs
    touch = 1 if touchscreen == 'Yes' else 0
    ips_disp = 1 if ips == 'Yes' else 0
    
    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Form the prediction array
    # Order: Company, TypeName, Ram, Weight, Touchscreen, Ips, ppi, Cpu brand, HDD, SSD, Gpu brand, os
    import pandas as pd
    input_df = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touch,
        'Ips': ips_disp,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }])

    # Predict
    predicted_log_price = pipe.predict(input_df)[0]
    actual_price = int(np.exp(predicted_log_price))
    
    st.success(f"### Estimated Price: ₹ {actual_price:,}")
