import os
import streamlit as st
import streamlit.components.v1 as components



st.title("Digital Twin Case Study: Monitoring Dam Vibrations to Optimize Maintenance Cycles")

st.markdown(''' This blog presents a case-study on how historical operational data can be used to understand the state of vibrations.''')

col1, col2 = st.columns([0.5 , 1])
with col1:
    st.image("media/car.jpg")

with col2:
    st.markdown(''' Let's consider a simple example - imagine buying a brand-new car having mileage of 20 miles per gallon. But over time, mileage can drop — perhaps to 15 miles per gallon after six months. And if the car isn’t maintained, more serious issues can arise, leading to unexpected breakdowns. This is why regular maintenance is essential — to keep the car running efficiently and avoid costly repairs.''')

st.markdown('''The same principle applies to our hydro turbines. When a turbine is first installed, it operates at peak efficiency. However, as it continues to run, wear and tear cause its performance to decline. **Preventive maintenance** is often scheduled to extend the life of the machinery and prevent catastrophic failures. This type of maintenance is done at regular intervals to ensure the turbine operates smoothly. 

However, there are two significant problems with the preventive maintenance approach:

1. **Unnecessary Maintenance:** Sometimes, maintenance is performed when it’s not needed. Returning to the car analogy, the car probably doesn’t need a 6-month check-up if it has not been used much. Similarly, if a hydro turbine hasn’t been operating frequently, shutting down the plant for maintenance could prove expensive.")

2. **Missed Maintenance:** On the other hand, there are times when maintenance is needed before the scheduled time. For example, a turbine might start showing signs of wear before its next planned maintenance. If this issue is not addressed, it could lead to a breakdown, resulting in costly and disruptive repairs.

In hydroelectric stations, monitoring vibrations can provide crucial insights into the turbine’s condition, allowing a shift from preventive maintenance to a more effective **predictive maintenance** strategy. 

This case study explores how this can be achieved.

### Our Approach

We use historical data from turbine operations to calculate the rate at which vibrations change over time. With this information, we will be able to understand the turbine’s current state and even estimate its future state. By the end of article, you will know what “understanding the current state” and “estimating the future state” mean.

### Data at a Glance

The turbine is equipped with sensors that record vibrations every second. We have data for three years: 2021, 2022 and 2023. A single day has 86,400 seconds, resulting in 86.4k data points per day. For three years, this amounts to approximately 96 million data points. This is a vast dataset, so to ease the analysis, we subsample the data every 5 minutes. The resampled data might look something like this:''')

st.image("media/1.png", caption="Fig 1. Scatter plot of vibration in turbine for years 2021–2023")

st.markdown('''In its current form, this graph doesn’t tell us much beyond the fact that vibrations range from 0 to 25 mils (1 mil = 1/1000 inch). We need to transform this graph using domain knowledge to reveal useful patterns.")
### Domain Knowledge"

- DK1: The turbine operates in three ON modes — Pump, Generate, and Condense. For our turbine, these modes are defined as follows:")
    - Pump: Power < -15 MW")
    - Generate: Power >=10 MW")
    - Condense: -2<Power <0 MW, rpm>=50")
    
- DK2: It takes about 1 minute for the turbine to reach any of the ON modes from a completely OFF state. Similarly, it takes about 30–40 seconds for the turbine to transition between ON modes.
- DK3: The turbine is considered to be in transition when its rotational speed is less than 90% of the maximum speed."

These three basic pieces of information are crucial for directing our analysis. The goal is to select a subset of points from the original data shown in Fig-1 that are similar to each other, or **homogenous**. Conceptually, vibrations during similar conditions should be the same and, if they’re not, wear and tear could be a factor responsible for that deviation.")

### Data Mining: Selecting Homogeneous Points

1. Selecting timeframe: Let’s choose years 2021 and 2022 to perform analysis and develop a mathematical model.
2. Selecting a Mode: Points corresponding to the ‘Generate’ mode are selected (..... from DK1).
3. Eliminating Transients: Select points only when the plant has been in ‘Generate’ mode for at least 5 minutes to remove transient points (..... from DK2).
4. Rotational Speed Filtering: Remove points when the turbine has not reached ~90% of its maximum speed (..... from DK3).

Now, instead of creating a scatter plot like Fig-1, we will plot running time on the x-axis and vibration values on the y-axis. This plot will show vibration values recorded every 5 minutes for all instances when the plant was operating in the ‘Generate’ mode.''')

st.image("media/2.png", caption="Fig 2. Scatter plot of Vibration at different running times (Scale of x-axis is 5 mins. So running_time=50 actually means turbine has been running for 50x5 mins = 250 mins)")

st.markdown('''This new plot has less data noise and clearly shows two clusters. But why two? Upon investigation, it turns out the upper cluster corresponds to data points from 2021, while the lower cluster corresponds to data points from 2022 as shown below in Fig 3. But why did the turbine exhibit significantly lower vibrations in 2022 compared to 2021? It meant that maintenance must have happened in early 2022, resulting in the turbine's better functioning. [This conclusion is indeed correct. ...Verified]''')

st.image("media/3.png", caption="Fig 3.Plot shows 2 clusters corresponding to years 2021 and 2022")

st.markdown(" Since turbine is not shutdown too frequently within a year for maintenance, it is safe to assume that for year 2021, as time progressed the turbine’s performance must have decined. Right? If it did, how do we verify and calculate this. For this, let's color the points by quarters in year 2021. The resulting plot is shown in Fig 4.")

st.image("media/4.png", caption="Fig 4.Vibration values for year 2021 colored by Quarters, revealing an upward trend")


st.markdown('''As seen, vibrations progressively rise from Q1 to Q2 to Q3 to Q4. Hence, our assumption about declining turbine's performance turned out to be CORRECT. With the above plot, we transformed the non-informative graph in Fig-1 into a highly insightful visual. 

What about 2022? Does the same pattern hold?''')

st.image("media/5.png", caption="Fig 5.Vibration values for year 2022 colored by Quarters, revealing an upward trend")

st.markdown('''Yes, it does."
### Next Steps"

Using statistical and parametric analysis, we can determine the rate of change of vibration over time. This equation will allow us to understand the current state and estimate the future state of the turbine:

1. **Current State**: If current vibrations are higher than expected, the turbine is overstressed and requires maintenance, helping prevent **reactive maintenance.**

2. **Future State**: If current vibrations are lower than expected, scheduled maintenance can be delayed, preventing unnecessary shutdowns and optimizing the maintenance schedule.

There’s much more we can accomplish based on this information, such as refining the model with multivariate analysis, performing fault isolation (root-cause analysis), anomaly detection, and more."

The mathematical modeling of vibration will soon be updated ....''')











