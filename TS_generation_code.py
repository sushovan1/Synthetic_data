# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 19:50:08 2022

@author: CSU5KOR
"""

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
import base64


def convert_df(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

def generate_data(data_point_count=100,noise=100,add_trend=0):
    T=data_point_count
    x=np.arange(0,data_point_count)
    y=  np.sin(4*np.pi*x/T)+np.cos(8*np.pi*x/T)
    noise_val=noise+np.random.randn(data_point_count)
    if add_trend==1:
        z=(3+2*x)+y+noise_val
    if add_trend==0:
        z=y+noise_val
    return z

def generate_anomaly(data_point_count=100,noise=100,add_trend=0,point_anomaly=1,anomaly_percentage=10):
    ano_fac=anomaly_percentage/100
    y=generate_data(data_point_count,noise,add_trend)
    mean_y=np.mean(y)
    standard_deviation=np.std(y)
    anomaly_val=mean_y+6*standard_deviation
    anomaly_val_low=mean_y-6*standard_deviation
    if point_anomaly==1:
        indices=np.random.randint(0,data_point_count,int(data_point_count*ano_fac))
        random_val=np.random.randint(0,2,1)[0]
        if random_val==0:
            y[indices]=anomaly_val_low
        else:
            y[indices]=anomaly_val
    else:
        index=np.random.randint(0,data_point_count-20,1)[0]
        indices=np.arange(index,index+20)
        random_val=np.random.randint(0,2,1)[0]
        if random_val==0:
            y[indices]=anomaly_val_low
        else:
            y[indices]=anomaly_val
    return y,indices
        
############################################################################################
st.title('A simple time series generation app')
anomaly_flag=st.sidebar.radio('Choose data type',options=['Anomalous','Normal'],index=1)
if anomaly_flag=="Normal":
    data_count=st.sidebar.slider("choose count of data points",min_value=100,max_value=10000,value=500,step=100)
    trend_flag=st.sidebar.radio('Add trend',options=[0,1])
    noise_param=st.sidebar.slider("choose noise level",min_value=0,max_value=1000,value=100,step=50)
    generated_array=generate_data(data_point_count=data_count,noise=noise_param,add_trend=trend_flag)
    df=pd.DataFrame(generated_array)
    
    csv=convert_df(df)
    st.download_button("Download your data",csv, anomaly_flag+".csv",key="download-csv")
    
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=np.arange(0,data_count), y=generated_array,
                        mode='lines'))
    fig.update_layout(title="Normal data")
    #fig.write_html('first_figure.html', auto_open=True)
    #fig.show()
    st.plotly_chart(fig, use_container_width=False)
else:
    data_count=st.sidebar.slider("choose count of data points",min_value=100,max_value=10000,value=500,step=100)
    anomaly_count=st.sidebar.slider("choose percentage of anomaly points",min_value=2,max_value=100,value=10,step=2)
    trend_flag=st.sidebar.radio('Add trend',options=[0,1])
    noise_param=st.sidebar.slider("choose noise level",min_value=0,max_value=1000,value=100,step=50)
    anomaly_type=st.sidebar.radio('choose anomaly type',options=['Point','Persistent'])
    
    if anomaly_type=='Point':
        generated_array,indices=generate_anomaly(data_point_count=data_count,noise=noise_param,add_trend=trend_flag,point_anomaly=1,anomaly_percentage=anomaly_count)
        df=pd.DataFrame(generated_array)
        
        
    else:
        generated_array,indices=generate_anomaly(data_point_count=data_count,noise=noise_param,add_trend=trend_flag,point_anomaly=0,anomaly_percentage=anomaly_count)
        df=pd.DataFrame(generated_array)

    csv=convert_df(df)
    st.download_button("Download your data",csv, anomaly_flag+".csv",key="download-csv")
    
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=np.arange(0,data_count), y=generated_array,
                        mode='lines',name='Generated data'))
    fig.add_trace(go.Scatter(x=indices, y=generated_array[indices],
                        mode='markers',name='Anomaly points'))
    fig.update_layout(title="Anomaly data")
    #fig.write_html('first_figure.html', auto_open=True)
    #fig.show()
    st.plotly_chart(fig, use_container_width=False)
    

        
