import os
import streamlit as st
import pandas as pd


st.title("Appendix")
st.markdown('''
            The app is originally built for data analytics of IoT sensors for a Hydro Power Station. The data is proprietary (private), hence not available with the app. Although, a sample data is provided with the app for sake of utility.

            Columns in dataframe represent names of sensors (starting with 'HYDRO') and rest are engineered features.

            Engineered features include some time-based columns like year, month, day, etc and some domain-based features like running_time, mode, etc.

            In order to use app on custom data, feel free to modify the app.
            
            ''')


st.markdown(
    '''
    ------------------------------------------------------------------------------------
    :man-raising-hand: Created by [Mohit Gupta](https://mgupta70.github.io), Ph.D., Arizona State University

    :email: Contact me at : mgupta70@asu.edu
    '''
)