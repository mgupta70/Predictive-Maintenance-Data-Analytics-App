data_streaming_plot_description = ''' 

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
        '''
#####################################################################################
yoy_plot_description = '''  
- **How can this plot be useful :question:**
    - Look out for signs of deteriorating system health or abnormal operating conditions. For e.g., is the current month's vibration abnormally higher than past years'? 
    
To help quantify the state of the selected sensor at a :blue[macro level], two metrics are presented below the figure. 
1. **Current monthly average** of the selected sensor in :black[Black]. Exactly underneath this, another number will be present either in :red[Red] or :green[Green], represnting change in current month average w.r.t past years' monthly average.  
2. **Number of days in this month and year that witnessed highest daily average w.r.t to past years**.

**How can these metrics be useful :question:**
- For safe operation, the change should not be too high &  should be small.
    '''
######################################################################################
mom_plot_description = ''' 
        
Paramter **n** = Number of past months for comparison.
        
- **How can this plot be useful :question:**
    - Look out for signs of deteriorating system health. For e.g., is the current month's vibration abnormally higher than past n months? 

Same as Plot-2:
To help quantify the state of the selected sensor at a :blue[macro level], two metrics are presented below the figure. 
1. **Current monthly average** of the selected sensor in :black[Black]. Exactly underneath this, another number will be present either in :red[Red] or :green[Green], represnting change in current month average w.r.t past n months.  
2. **Number of days in this month that witnessed highest daily average w.r.t to past n months**. 
    '''
#########################################################################################
psuedo_sensor_plot_description = ''' 
- **What is Psuedo-sensor  :question:**

    - In Psuedo-Sensor analysis, selected sensor values are predicted by looking at values of other unrelated sensors.

- **How can this plot be useful :question:**
    - Ideally, the predicted values should be very close to true values. Hence, large deviation from central black line y=x may indicate potential Anomalies.' 
    '''
#########################################################################################
correlation_plot_description = '''  
- For ex: all 4 sensors listed below measures the air cooler temperature in Deg F.
    1. HYDRO-ABC-TE-001-F
    2. HYDRO-ABC-TE-002-F
    3. HYDRO-ABC-TE-003-F
    4. HYDRO-ABC-TE-004-F
    
- How can this plot be useful?
    - Ideally, the correlation of selected sensor with each of its same-family sensor should be high. If a sudden drop in correlation is observed it can point towards anomalies like miscalibration or a faulty sensor or loose-connections (if it happens).' 
    '''