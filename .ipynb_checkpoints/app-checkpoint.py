import streamlit as st
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import datetime
import numpy as np
warnings.filterwarnings('ignore')

import seaborn as sns
import sys
sys.path.append(os.path.abspath('..'))
import datetime as dt
from utils.helpers import *
from utils.utils import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

import streamlit.components.v1 as components
from multiprocessing import Process

# Ideas:
# caching so that everything does not run
# plot-1: daily trend line
# plot-2: % days experiencing more vibrations than 2 years. : Plot them sequentially instead of year-wise
# plot-3: Move n_months close to header
# plot-4: updated plots (later with selection criterias) | plotly
# plot-5: monthly, weekly


# Page config

st.set_page_config(page_title="HPP Digital Twins ASU-SRP", page_icon=":bar_chart:",layout="wide")
st.title(" :bar_chart: Predictive Analytics for MF2")
st.markdown('<style>div.block-container{padding-top:3rem;}</style>',unsafe_allow_html=True)

# st.sidebar.image('media/asu_logo.png')
# st.logo('media/srp_logo.png')

st.logo('media/asu_logo.png')
st.sidebar.image('media/srp_logo.png')

# Data
#fpth = 'data_checkpoints/yearly_dfs_clean/iot_all_FE.pkl'
fpth = 'data/iot_all_FE2_5min.pkl'

@st.cache_data(show_spinner = 'Loading data...')
def load_data(fpth):
    df = pd.read_pickle(fpth)
    return df
df = load_data(fpth)
sensors_list = [o for o in list(df.columns) if 'HYDRO' in o]

#sidebar
sensor = st.sidebar.multiselect("Select Sensor(s)", sensors_list)
month_name = st.sidebar.selectbox("Month", list(month_num2name.values()), index=0, key=21)  # default - January 
year_num = st.sidebar.selectbox("Select Year", [int(yr) for yr in list(df.year.unique())], index=3, key=31)
n_months = st.sidebar.selectbox("Past n Months", list(range(1, 12)), index=1, key=32)

if sensor:
    in_family, out_family = get_same_family_sensors(sensors_list, sensor=sensor[0])


#####################
# Advanced Analytics
#####################

modify = st.checkbox("Add filters")
if not modify:
    filtered_df = df.copy()
else:
    col1, col2 = st.columns((2))  
    startDate = df.index.min() 
    endDate = df.index.max()  
    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))
    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))
    if date1 and date2:
        df = df[(df.index >= date1) & (df.index<= date2)].copy()
    filtered_df = filter_dataframe(df)


    
##########
# Plot-1
##########
col11, col12 = st.columns([1 , 0.01])
with col11: 
    st.subheader('1. Data Streaming')
    with st.expander('Select a sensor to analyse. For more information - Click here'):
        st.markdown(''' 

##### Top plot

- Points in :red[Red] are real (actual) values observed for the sensor you selected.
- Points in :green [Green] are predicted values for the _selected sensor_.
- **How can this plot be useful :question:**
    - Select multiple sensors of same family for e.g. - *HYDRO_M2-GEN-HTX-TE-001-F.MC@MF* & *HYDRO_M2-GEN-HTX-TE-002-F.MC@MF* . Ideally, you can expect their curves to have similar shapes or even overlap. If they don't, it might reflect a potential problem like :blue-background[loose connection/ miscalibration/ faulty sensor]. 
 
##### Bottom plot
- Bottom plot shows the change in real (actual) values in compared to previous timestep ( which is 10 seconds in our case). In simple words, it is a plot of:

$$
\t{Sensor}(t) - \t{Sensor}(t - \Delta t) \quad \t{where} \quad \Delta t = 10 \t{ seconds}
$$

- How can this plot be useful :question:**
    - Sudden large changes in sensor values i.e., sudden peaks in the graph may hint towards :blue-background[some out-of-normal behavior]. However, these 'sudden' peaks occur when plant is transitioned from one mode to another. For e.g., when plant is turned into *condense* mode after *generate* mode, peaks in *power-output* sensor could be observed. In such cases, these peaks don't reflect any anomalous behaviour and will eventually fade away. **Hence, it is a recommendation to analyse the peaks when plant is in steady-state and not transitioning between different modes of operation**. 
        ''')
    if sensor:
        df1 = df[sensor].copy()
#         df1 = df1.head(10000).copy()
        ## Top plot and Bottom plot
        fig1 = plot_sensor_data(df1, sensor, is_app=True)
        st.plotly_chart(fig1,use_container_width=True)
    

    
    
    
# ##########
# # Plot-2
# ##########
col21, col22 = st.columns([1 , 1])
with col21:
    st.subheader('2. Year-on-Year Trend')
    with st.expander('This plot helps us see the current state of the selected sensor w.r.t. past years. For more information - Click here'):
        st.markdown('''  
- **How can this plot be useful :question:**
    - Look out for signs of deteriorating system health or abnormal operating conditions. For e.g., is the current month's vibration abnormally higher than past years'? 
    
To help quantify the state of the selected sensor at a :blue[macro level], two metrics are presented below the figure. 
1. **Current monthly average** of the selected sensor in :black[Black]. Exactly underneath this, another number will be present either in :red[Red] or :green[Green], represnting change in current month average w.r.t past years' monthly average.  
2. **Number of days in this month and year that witnessed highest daily average w.r.t to past years**.

**How can these metrics be useful :question:**
- For safe operation, the change should not be too high &  should be small.
    ''')
    if sensor and month_name:        
        ## Plotly
        fig2, avg_data = plotly_YOY_trend(df, sensor, month_name)
        st.plotly_chart(fig2,use_container_width=True)
        
        monthly_averages = {}
        for yr in avg_data.year.unique():
            monthly_averages[yr] = avg_data.loc[yr][sensor[0]].mean()
            
        m1_value = get_m1(monthly_averages, year_num)
        m2_value = get_m2(monthly_averages, year_num)
        m3_value = get_m3(avg_data, year_num, sensor)
               
        col_left, col_right = st.columns(2)
    
        with col_left:
            st.metric(label='Monthly average', value=f'{m1_value} {sensors_units[sensor[0]]}', delta=f'{m2_value} {sensors_units[sensor[0]]} w.r.t. past yrs')

        with col_right:
            st.metric(label='Number of days with Peak daily Average', value=f'{m3_value} Days', delta=None)
 
    
##########
# Plot-3
##########
with col22:
    st.subheader('3. Month-on-Month Trend')
    with st.expander('This plot helps us see the current state of the selected sensor w.r.t. past months. For more information - Click here'):
        st.markdown(''' 
        
Paramter **n** = Number of past months for comparison.
        
- **How can this plot be useful :question:**
    - Look out for signs of deteriorating system health. For e.g., is the current month's vibration abnormally higher than past n months? 

Same as Plot-2:
To help quantify the state of the selected sensor at a :blue[macro level], two metrics are presented below the figure. 
1. **Current monthly average** of the selected sensor in :black[Black]. Exactly underneath this, another number will be present either in :red[Red] or :green[Green], represnting change in current month average w.r.t past n months.  
2. **Number of days in this month that witnessed highest daily average w.r.t to past n months**. 
    ''')
    if sensor and year_num:        
        ## Plotly
        fig3, df3 = plotly_MOM_trend(df, sensor, month_name, year_num, n_months)
        st.plotly_chart(fig3,use_container_width=True)
        
        avg_data2 = df3.groupby(['month', 'day'])[['day', 'month', 'year', sensor[0]]].mean()
        
        m4_value, m5_value = get_m4_m5(df3, month_name, sensor) 
        m6_value = get_m6(avg_data2, month_name, sensor)
        
        col_left, col_right = st.columns(2)
    
        with col_left:
            st.metric(label='Monthly average', value=f'{m4_value} {sensors_units[sensor[0]]}', delta=f'{m5_value} {sensors_units[sensor[0]]}')

        with col_right:
            st.metric(label='Daily average exceeds', value=f'{m6_value} Days', delta=None)


    
    
###########
## Plot-4
##########
col4, col5 = st.columns([1, 1])
with col4:
    st.subheader('4. Psuedo-Sensor Analysis')
    with st.expander('Plot shows the true value on x-axis and corresponding predicted value on y-axis. For more information - Click here'):
        st.markdown(''' 
- **What is Psuedo-sensor  :question:**

    - In Psuedo-Sensor analysis, selected sensor values are predicted by looking at values of other unrelated sensors.

- **How can this plot be useful :question:**
    - Ideally, the predicted values should be very close to true values. Hence, large deviation from central black line y=x may indicate potential Anomalies.' 
    ''')
    # if sensor:
    #     conditions = [lambda df: df['mode'].isin(['pump', 'gen', 'condense']),]
    #     df4 = filter_df(df, date1 = '2023-01-01', date2 = '2024-01-01', conditions=conditions) #######################
    #     # Plotly
    #     fig4 = PsuedoSensorAnalysis(df=df4, indep=sensor[0], deps=in_family, exclude=None, fit_model='Linear')
    #     st.plotly_chart(fig4,use_container_width=True)


##########
# Plot-5
##########
with col5:
    st.subheader('5. Correlation Analysis')
    with st.expander('Plotting monthly correlation of selected sensor with same-family sensors. For more information - Click here'):
        st.write('''  
- For ex: all 4 sensors listed below measures the air cooler temperature in Deg F.
    1. HYDRO_M2-GEN-HTX-TE-001-F.MC@MF
    2. HYDRO_M2-GEN-HTX-TE-002-F.MC@MF
    3. HYDRO_M2-GEN-HTX-TE-003-F.MC@MF
    4. HYDRO_M2-GEN-HTX-TE-004-F.MC@MF
    
- How can this plot be useful?
    - Ideally, the correlation of selected sensor with each of its same-family sensor should be high. If a sudden drop in correlation is observed it can point towards anomalies like miscalibration or a faulty sensor or loose-connections (if it happens).' 
    ''')

    if sensor:
        
        df5 = df.copy()
        time_scale = 'monthly' 
        correlations_dict = {}   
        for s in in_family:
            if s not in correlations_dict:
                correlations_dict[s] = []

        if time_scale == 'monthly':
            for yr in df5.index.year.unique():
                data = df5[df5.index.year==yr]
                for m in list(range(1,13)):
                    sub_df = data[data.index.month==m]
                    for s in in_family:
                        if len(sub_df)>0:
                            corr = sub_df[s].corr(sub_df[sensor[0]])
                            correlations_dict[s].append(corr)
                        else:
                            correlations_dict[s].append(None)

        xticks = []
        for yr in df.index.year.unique():
            for m in list(range(1,13)):
                xticks.append(f"{yr}-{month_num2name[m]}")

        rotation = 90
        fig = go.Figure()
        for s, v in correlations_dict.items():
            v = v[9:-3]
            x_vals = xticks[9:-3]
            y_vals = v

            # Add scatter plot for each sensor
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=s))

            # Highlight None values with red "x"
            for i, value in enumerate(y_vals):
                if value is None:
                    fig.add_trace(go.Scatter(
                        x=[x_vals[i]],
                        y=[0.0],
                        mode='markers',
                        marker=dict(color='red', symbol='x'),
                        showlegend=False
                    ))

        # Update layout
        fig.update_layout(
            title='Correlation Analysis',
            xaxis_title='Yr-Month',
            yaxis_title='Correlation',
            xaxis=dict(tickmode='array', tickvals=x_vals, tickangle=rotation),
            yaxis=dict(showgrid=True),
            legend=dict(x=0.4, y=0.2, bgcolor = 'rgba(255,255,255,0.5)'),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

##########
# Plot-6
##########
st.subheader('6. Heatmap of Sensor Data')
with st.expander("This plot gives a snapshot of sensor's behavior in its lifetime. For more information - Click here"):
    st.markdown(''' 

##### To Do
Details about what they see on hovering
Check - missing days verification

        ''')
if sensor:
    df_gg = get_pivot_table(filtered_df, sensor)
    fig6 = plot_daily_heatmap(df_gg, sensor)
    st.plotly_chart(fig6,use_container_width=True)

