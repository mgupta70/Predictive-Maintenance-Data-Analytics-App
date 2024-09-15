import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from fastai.vision.all import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
import plotly.express as px
import sys
sys.path.append(os.path.abspath('..'))
from utils.helpers import *
import streamlit as st
import datetime as dt

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

import streamlit.components.v1 as components
from multiprocessing import Process


@st.cache_data(show_spinner=False)
def add_modes_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'mode' column to the dataframe based on specific conditions 
    involving RPM and power variables.
    """
    df = dataframe.copy()
    var_rpm1 = 'HYDRO_M2-GOV-SI-001-SI.MC@MF'
    var_rpm2 = 'HYDRO_M2-GOV-SI-002-SI.MC@MF'
    var_pwr = 'HYDRO_M2-DPM-GEN-MW.MC@MF'
    
    df['mode'] = np.where((df[var_rpm1] == 0.0) | (df[var_rpm2] == 0.0), 'off', 
                  np.where(df[var_pwr] <= -15, 'pump',
                           np.where(df[var_pwr] >= 10, 'gen',
                                    np.where((df[var_pwr] > -2) & 
                                             (df[var_pwr] < 0) & 
                                             ((df[var_rpm1] > 50.0) | 
                                             (df[var_rpm2] > 50.0)), 
                                             'condense', 'transition'))))
    return df


@st.cache_data
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common time-based features to the dataframe based on the datetime index.
    """
    time_features = ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofyear', 'dayofweek', 'quarter', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_leap_year']
    
    for feature in time_features:
        df[feature] = getattr(df.index, feature)
        
    df['week'] = np.ceil(df['dayofyear'] / 7).astype(int)
    
    return df


@st.cache_data(show_spinner=False)
def get_df_bw_time(df: pd.DataFrame, start_date: str, end_date: str, include_last_day: bool = False) -> pd.DataFrame:
    """
    Retrieves a subset of the dataframe between two dates.
    ex: date1 =  '2020-11-09'
    """
    start_date = pd.to_datetime(f'{start_date} 00:00:00')
    end_date = pd.to_datetime(f'{end_date} 23:59:59' if include_last_day else f'{end_date} 00:00:00')
    
    return df[start_date:end_date].copy()
    
@st.cache_data(show_spinner=False)
def add_running_time_column(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """
    Adds a 'running_time' column to the dataframe which tracks the duration of each mode.
    """
    running_time = []
    temp_mode = 'idk'
    temp_time = 1
    for ind in range(len(df)):
        current_mode = df['mode'].iloc[ind]
        if current_mode == temp_mode:
            temp_time = temp_time + 1
            running_time.append(temp_time)
        elif current_mode != temp_mode:
            temp_mode = current_mode
            temp_time = 1
            running_time.append(temp_time)
    
    df['running_time'] = running_time
    
    if save:
        print('saving data')
        save_pickle('../data_checkpoints/df_running_time999.pkl', df)
    return df
    
@st.cache_data(show_spinner=False)
def get_same_family_sensors(sensors_list: list, sensor: str) -> tuple[list, list]:
    """6
    Identifies sensors from the same family based on the provided sensor string.
    """
    if '-' in sensor:
        match_pattern = '-'.join(sensor.split('-')[:-2])
    else:
        match_pattern = '_'.join(sensor.split('_')[:-2])
        
    in_family = [o for o in sensors_list if match_pattern in o]
    in_family.remove(sensor)
    out_family = list(set(sensors_list).difference(set(in_family)))
    
    return in_family, out_family
        




##########
# Plot-1
##########
@st.cache_data
def plot_sensor_data(df, sensor, is_app=True):
    
    delta = df[sensor].diff()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=(f'{sensor[0]}', 'Delta plot'), row_heights=[0.7, 0.3])
    
    # Add First plot - Sensor values
    fig.add_trace(go.Scatter(x = df.index, y=df[sensor[0]], mode = 'lines', name = 'actual', line=dict(color='red')), row=1, col=1)
    
    if len(sensor)>1:
        for idx in range(1, len(sensor)):
            fig.add_trace(go.Scatter(x = df.index, y=df[sensor[idx]], mode = 'lines', name = f'{sensor[idx]}', ), row=1, col=1)
        
        
    fig.update_yaxes(title_text = f'{sensors_units[sensor[0]]}', row=1, col=1)
    fig.update_layout(legend = dict(x = 0.01, y = 0.99, bgcolor = 'rgba(255,255,255,0.5)'))

    # Add second plot - Change in Sensor values
    fig.add_trace(go.Scatter(x=delta.index, y=delta[sensor[0]], mode='lines', name='Change in Sensor Values', line=dict(color='blue')), row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="Sensor Data and Changes Over Time")
    fig.update_yaxes(title_text = f'{sensors_units[sensor[0]]}', row=2, col=1)
    fig.update_xaxes(title_text = 'Time', row=2, col=1)
    fig.update_layout(legend = dict(x = 0.01, y = 0.99, bgcolor = 'rgba(255,255,255,0.5)'))
    
    if is_app:
        return fig
    else:
        fig.show()
    


##########
# Plot-2
##########
@st.cache_data
def get_m1(monthly_averages, year_num):
    return round(monthly_averages[year_num],1)
    
@st.cache_data
def get_m2(monthly_averages, year_num):
    total = 0
    count = 0
    for yr in monthly_averages:
        if yr < year_num:
            total+=monthly_averages[yr]
            count+=1
    if count>0:
        prev_yr_avg = total/count
        m2 = round((monthly_averages[year_num] - prev_yr_avg)*100/prev_yr_avg,1)
    else:
        m2=0
    return m2
    
@st.cache_data
def get_m3(avg_data, year_num, sensor):
    
    prev_years = []
    for yr in avg_data.year.unique():
        if yr < year_num:
            prev_years.append(yr)

    if len(prev_years)>0:
        baseline_max_daily = avg_data.loc[prev_years].groupby(level='day')[sensor[0]].max()
        current_avg_daily = avg_data.loc[year_num][sensor[0]]

        if len(baseline_max_daily)==len(current_avg_daily):
            exceeded_days = (current_avg_daily > baseline_max_daily).sum()
        else:
            common_days = baseline_max_daily.index.intersection(current_avg_daily.index)
            baseline_max_daily_common = baseline_max_daily.loc[common_days]
            current_avg_daily_common = current_avg_daily.loc[common_days]

            exceeded_days = (current_avg_daily_common > baseline_max_daily_common).sum()

    else:
        exceeded_days = 0
    return exceeded_days

@st.cache_data
def plot_YOY_trend(df, sensor, month_name,  fig_sz, label_sz, line_wd, rotation, is_app=True):
    
    ####### global #########################
    years = [2020, 2021, 2022, 2023] # df.index.year.unique()
    colors = sns.color_palette('husl', n_colors=len(years))
    cmap = dict(zip(years, colors))
    #######################################
    
    month_num = month_name2num[month_name]
    df2 = df[df['month'] == month_num].copy()
    label_sz = 25
    fig, ax = plt.subplots(1, 1, figsize=fig_sz)
    ax.tick_params(axis="x", labelsize=label_sz, rotation = rotation)
    ax.tick_params(axis="y", labelsize=label_sz)
    for year in df2.year.unique():
        year_data = df2[df2.year == year]
        sns.lineplot(x=year_data.day, y=year_data[sensor[0]],
                     err_style=None, hue=year, palette=cmap, ax=ax, linewidth=line_wd)

        ax.set_xlim(0, 33)
        ax.set_xticks(range(1, 32, 5))  # Set x-axis ticks at intervals of 5 days
        ax.set_title(f'Month: {month_num2name[month_num]}', fontsize=label_sz)
        ax.set_xlabel('Days', fontsize=label_sz)
        ax.set_ylabel(f'{sensors_units[sensor[0]]}', fontsize=label_sz)
        ax.legend(title = 'Year', loc='upper left', title_fontsize = label_sz, fontsize=label_sz)
    if is_app:
        return fig
    else:
        plt.show()
        

@st.cache_data      
def plotly_YOY_trend(df, sensor, month_name):
    
    custom_colors = {2020: px.colors.qualitative.Plotly[0], 
                     2021: px.colors.qualitative.Plotly[1], 
                     2022: px.colors.qualitative.Plotly[2], 
                     2023: px.colors.qualitative.Plotly[3]}

    # Convert month name to number
    month_num = month_name2num[month_name]

    # Filter the dataframe for the selected month
    df2 = df[df['month'] == month_num].copy()
    
    avg_data = df2.groupby(['year', 'day'])[['day', 'year', sensor[0]]].mean()

    # Create the line plot using Plotly Express
    fig = px.line(
        avg_data,
        x='day',
        y=sensor[0],
        color='year',
        color_discrete_map=custom_colors,  # Apply custom colors
        markers=True,
        title=f'Month - {month_name}', height=500, width = 1000
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Day of Month',
        yaxis_title=sensors_units[sensor[0]],
        template='plotly_white'
    )
    
    return fig, avg_data

##########
# Plot-3
##########
@st.cache_data
def get_m4_m5(df, month_name, sensor):
    ''' current month average '''
    month_num = month_name2num[month_name]
    m4 = 0
    prev_mnth_avg = []
    for mnth in df['month'].unique():
        if mnth == month_num:
            month_df = df[df['month']==month_num]
            m4 = round(month_df[sensor[0]].mean(),1)
        else:
            #print('prev')
            month_df = df[df['month']==mnth]
            mean_value = round(month_df[sensor[0]].mean(),1)
            prev_mnth_avg.append(mean_value)
   
    m5 = round((m4 -  np.mean(prev_mnth_avg))*100/np.mean(prev_mnth_avg),1)
    return m4, m5

@st.cache_data
def get_m6(avg_data, month_name, sensor):
    month_num = month_name2num[month_name]
    prev_mnths = []
    for mnth in avg_data.month.unique():
        if mnth != month_num:
            prev_mnths.append(mnth)

    if len(prev_mnths)>0:
        baseline_max_daily = avg_data.loc[prev_mnths].groupby(level='day')[sensor[0]].max()
        current_avg_daily = avg_data.loc[month_num][sensor[0]]

        if len(baseline_max_daily)==len(current_avg_daily):
            exceeded_days = (current_avg_daily > baseline_max_daily).sum()
        else:
            common_days = baseline_max_daily.index.intersection(current_avg_daily.index)
            baseline_max_daily_common = baseline_max_daily.loc[common_days]
            current_avg_daily_common = current_avg_daily.loc[common_days]

            exceeded_days = (current_avg_daily_common > baseline_max_daily_common).sum()

    else:
        exceeded_days = 0

    return exceeded_days


@st.cache_data
def plot_MOM_trend(df, sensor, month_name, year_num, n_months, fig_sz, label_sz, line_wd, rotation, is_app=True):
    
    month_num = month_name2num[month_name]
    end_date = dt.date(year_num, month_num, 1) + pd.offsets.MonthEnd(1)
    start_date = (end_date - pd.DateOffset(months=n_months)).replace(day=1)
    df3 = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    label_sz = label_sz
    fig, ax = plt.subplots(1, 1, figsize=fig_sz)
    ax.tick_params(axis="x", labelsize=label_sz, rotation = rotation)
    ax.tick_params(axis="y", labelsize=label_sz)
    sns.lineplot(x=df3.dayofyear, y=df3[sensor[0]],
                 err_style=None, ax=ax, linewidth=line_wd)
    ax.set_xlabel(' ', fontsize=label_sz)
    ax.set_title(f'{sensor[0]}', fontsize=label_sz)
    ax.set_ylabel(f'{sensors_units[sensor[0]]}', fontsize=label_sz)
    if is_app:
        return fig
    else:
        plt.show()

@st.cache_data
def plotly_MOM_trend(df, sensor, month_name, year_num, n_months):
    
    month_num = month_name2num[month_name]
    end_date = dt.date(year_num, month_num, 1) + pd.offsets.MonthEnd(1)
    start_date = (end_date - pd.DateOffset(months=n_months)).replace(day=1)
    df3 = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    
    avg_data = df3.groupby(['year', 'month', 'day'])[['year','month', 'day', sensor[0]]].mean()
    # Create a new column for combined 'month-day' information
    avg_data['month_day'] = avg_data['year'].astype(str) + '-' +  avg_data['month'].astype(str) + '-' + avg_data['day'].astype(str)

    # Create the line plot using Plotly Express
    fig = px.line(
        avg_data,
        x="month_day",
        y=sensor[0],
        markers=True,
        title=f"Timeframe: {month_num2name[start_date.month]}'{str(start_date.year)[-2:]} - {month_name}'{str(end_date.year)[-2:]}",  
        height=500, width = 1000
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Day',
        yaxis_title=sensors_units[sensor[0]],
        template='plotly_white'
    )
    return fig, df3



##########
# Plot-4
##########
@st.cache_data
def filter_df(dataframe, date1, date2, conditions=None):
    ''' 
    Inputs:
    1. dataframe
    2. date1 = start date
    3. date2 = end date
    4. conditions = list of conditions
    # Example:
        conditions = [
            lambda df: df['year'].isin([2020, 2021, 2022, 2023]),
            lambda df: df['month'].between(1, 12),
            lambda df: df['mode'].isin(['gen']),
            lambda df: df[var_pwr].between(10, 75)
        ]
    4. modes = list of pump, gen, condense, transition | if None -> means all
    '''
    df = get_df_bw_time(dataframe, date1, date2, include_last_day= False)
    
    filtered_df = df.copy()  # Start with a copy of the original DataFrame
    if conditions:
        for condition in conditions:
            filtered_df = filtered_df[condition(filtered_df)]
    return filtered_df

@st.cache_data
def PsuedoSensorAnalysis(df, indep, deps, exclude=None, fit_model = 'Linear'):
    '''
    deps = can be a list of columns
    indep = single variable (not a list)
    exclude = list of sensors (column names) to exclude as features
    '''
    
    print('Total features: ', len(deps))
    
    X = np.array(df.loc[:, deps])
    y = np.array(df.loc[:, indep])

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    if fit_model == 'Linear':
        model = LinearRegression(fit_intercept=True)   
        
    model.fit(X_scaled, y)
    y_preds = model.predict(X_scaled)

    print('MSE', mean_squared_error(y, y_preds))
    
    x_max = y.max()
    

    # plotly
    sorted_indices = np.argsort(y)
    y_sorted = y[sorted_indices]
    y_preds_sorted = y_preds[sorted_indices]
    
    print('plotting starts')
    # Initialize a FigureResampler object
    fig = FigureResampler(go.Figure())

    # Add scatter plot to the figure
    fig.add_trace(go.Scattergl(x=y_sorted, y=y_preds_sorted, mode='markers', marker=dict(size=2), name='Predicted vs Actual'))

    # Add ideal fit line
    fig.add_trace(go.Scattergl(x=[0, y_sorted[-1]], y=[0, y_sorted[-1]], mode='lines', line=dict(color='black'), name='Ideal Fit'))

    # Update layout
    fig.update_layout(
        xaxis_title='Actual',
        yaxis_title='Predicted',
        title='Predicted vs Actual',
        legend=dict(x=0, y=1)
    )

    # Show the figure with resampling enabled
    #fig.show_dash()
    return fig


##########
# Plot-6
##########
@st.cache_data
def add_yr_month_column(df):
    yr_month = []
    for ind in range(len(df)):
        temp = str(int(df['year'].iloc[ind])) + '-' + str(int(df['month'].iloc[ind]))
        yr_month.append(temp)
    df['year_month'] = yr_month
    return df

@st.cache_data
def drop_rows(df, n, from_top=True):
    if from_top:
        df.drop(df.head(n).index, inplace = True)
    else:
        df.drop(df.tail(n).index, inplace = True)
    return df

@st.cache_data
def get_empty_pivot_table(df):
    # prepare an empty df_gg with all year_months and days
    df_gg = pd.DataFrame(columns=['year_month', 'day'])
    # Loop through unique years and months
    for yr in df['year'].unique():
        for m in range(1,13):
            # Create a DataFrame for each combination of year and month
            temp_df = pd.DataFrame({
                'year_month': [f'{int(yr)}-{int(m)}'] * 31,
                'day': range(1, 32),
            })
            # Append the temporary DataFrame to df_gg
            df_gg = pd.concat([df_gg, temp_df], ignore_index=True)
    # Adjust according to available data
    drop_rows(df_gg, 279, from_top=True)
    drop_rows(df_gg, 63, from_top=False)
    df_gg['day'] = df_gg['day'].astype(float)
    return df_gg

@st.cache_data
def get_pivot_table(df, sensor):
    df = add_yr_month_column(df)
    df_grouped = df.groupby(['year_month', 'day']).agg({sensor[0]: 'mean'}).reset_index()
    df_gg = get_empty_pivot_table(df)
    values = []
    for ind in range(len(df_gg)):
        ym, d = df_gg.iloc[ind]['year_month'], df_gg.iloc[ind]['day']
        v = df_grouped[(df_grouped['year_month']==ym) & (df_grouped['day']==d)][sensor[0]]
        v = list(v)
        # print(ym, d, v)
        values.append(None if len(v)==0 else v[0])
        
    df_gg[sensor[0]] = values
    return df_gg

@st.cache_data
def plot_daily_heatmap(df_gg, sensor):
    data = [go.Heatmap(
        x=df_gg['year_month'],
        y=df_gg['day'],
        z=df_gg[sensor[0]],
        colorscale='Jet'
    )]

    layout = go.Layout(
        title=f'Average Daily {sensor[0]} Values Across Years and Months'
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig

######################
# Advanced Analytics
#######################
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # modify = st.checkbox("Add filters")

    # if not modify:
    #     return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    # for col in df.columns:
    #     if is_object_dtype(df[col]):
    #         try:
    #             df[col] = pd.to_datetime(df[col])
    #         except Exception:
    #             pass

        # if is_datetime64_any_dtype(df[col]):
        #     df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            # elif is_datetime64_any_dtype(df[column]):
            #     user_date_input = right.date_input(
            #         f"Values for {column}",
            #         value=(
            #             df[column].min(),
            #             df[column].max(),
            #         ),
            #     )
            #     if len(user_date_input) == 2:
            #         user_date_input = tuple(map(pd.to_datetime, user_date_input))
            #         start_date, end_date = user_date_input
            #         df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df
    