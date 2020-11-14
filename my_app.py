# streamlit run my_app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta
import altair as alt
import plotly.express as px
import plotly.graph_objs as go
import datetime
from PIL import Image

session = st.sidebar.selectbox("Section", ["Welcome!", "Overview", "By Country", "Economic Perspective", "Dietary Perspective"])
st.title('COVID-19 Dashboard')

if session == "Welcome!":

    st.sidebar.subheader("Welcome to our dashboard!")

    # image
    image = Image.open('img/covid-19.png')
    st.image(image, width = 700)
    st.subheader("Introduction")
    st.write("""
    This is a dashboard about COVID-19, its economic influence and the potential dietary method to prevent it. This dashboard is divided into the following five parts:
    - Welcome!
    - Overview
    - By Country
    - Economic Perspective
    - Dietary Perspective
    
    The codes and Machine Learning analysis can be found at the [Github repository](https://github.com/biostat823-FinalProject/Game-of-Data).""")
    

if session == "Overview":
    df_line = pd.read_csv('./Data/by_country.csv')[['location', 'date', 'total_cases', 'total_deaths']]
    df_line = df_line[df_line['location'] == 'World']
    date = df_line['date']
    df_line = df_line.rolling(3).mean()
    df_line['date'] = pd.to_datetime(date).dt.date
    df_line = df_line.dropna()
    df_line.rename({'total_cases': 'Total Cases', 'total_deaths': 'Total Deaths'}, axis = 1, inplace = True)
    
    #sidebar
    st.sidebar.subheader("Overview")
    # date_range = st.sidebar.date_input('Range of date:', [min(df_line['date']), max(df_line['date'])],min_value = min(df_line['date']), max_value = max(df_line['date']))
    date_range =  st.sidebar.slider('Range of date:', min(df_line['date']), max(df_line['date']), (pd.to_datetime('2020-03-01').date(), pd.to_datetime('2020-10-01').date()))
    cols = st.sidebar.multiselect('Measure of Severity:', df_line.columns[:2].to_list(), default = df_line.columns[:2].to_list())
    dates = [date_range[0]]
    while dates[-1] <= date_range[1]:
        dates.append(dates[-1] + timedelta(days = 1))
    df_line = df_line[df_line['date'].isin(dates)][cols + ['date']]

    #show graph
    fig_line = go.Figure()
    colors = ['rgba(255,127,14,1)', 'rgba(31,119,180,1)']
    
    for i in cols:
        df_line[i] = round(df_line[i], 0)
        fig_line.add_trace(go.Scatter(x = df_line['date'],
                                      y = df_line[i],
                                      marker = dict(color = colors[i == 'Total Cases']),
                                      mode = 'lines',
                                      name = i))
    fig_line.update_xaxes(title_text = 'Date', gridcolor = 'whitesmoke', linecolor = 'black')
    fig_line.update_layout({'plot_bgcolor': 'rgb(255,255,255)', 'paper_bgcolor': 'rgb(255,255,255)'})
    fig_line.update_yaxes(title_text = 'Total Number', gridcolor = 'whitesmoke', linecolor = 'black')

    # plot
    st.subheader('Total Number of COVID-19 Cases/Deaths by Date')
    st.plotly_chart(fig_line)
    
    with st.beta_expander("Figure Details"):
         st.write("""
            The data is from [Amazon Web Services (AWS) data lake](https://aws.amazon.com/blogs/big-data/a-public-data-lake-for-analysis-of-covid-19-data/). The cases and deaths for each day is calculated by the rolling mean with a window of 3 days.
         """)

if session == 'By Country':

    df_map = pd.read_csv('./Data/by_country.csv')
    df_map['date'] = pd.to_datetime(df_map['date']).dt.date
    df_map = df_map[~df_map['location'].isin(['World', 'International'])]

    # sidebar
    st.sidebar.subheader("By Country")
    check_map = st.sidebar.checkbox('Show Animation Plot', value = True)
    dates = st.sidebar.date_input('Date:', datetime.date(2020,11,1), min_value = min(df_map['date']), max_value = max(df_map['date']))
    method1 = st.sidebar.selectbox("Measure of Severity:", ['Cases', 'Deaths'])
    method2 = st.sidebar.selectbox("Calculation Method:", ['Total', 'New', 'Total per Million', 'New per Million'])
    num = st.sidebar.slider('Number of Top Countries:', 1, 100, (1, 5))
    # num = st.sidebar.number_input("Number of Top Countries:", 1, 20)
    # st.sidebar.button("Show", key=1)

    # graph
    df_map = df_map[df_map['date'] == dates]
    if len(method2.split()) > 1:
        col_name = method2.lower().split()[0] + '_' + method1.lower() + '_' + '_'.join(method2.lower().split()[1:])
    else:
        col_name = method2.lower() + '_' + method1.lower()
    fig_map = go.Figure(data = go.Choropleth(
        locations = df_map['location'],
        z = df_map[col_name].astype(float),
        locationmode = 'country names',
        colorscale = 'Reds',
        autocolorscale = False,
        marker_line_color = 'black',
        # text = '<b>' + dff['State'] + '</b><br>' + field + ' Field: ' + dff[field].astype(int).astype(str)
    ))
    
    fig_map.update_layout(
        width = 900, height = 500,
        geo = dict(showlakes = True, lakecolor = 'rgb(255, 255, 255)'))
    
    # animation plot
    if check_map:
        st.subheader('COVID-19 Cases from February to November at Country Level')
        html_string = """<iframe src='https://flo.uri.sh/visualisation/4299524/embed' title='Interactive or visual content' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/4299524/?utm_source=embed&utm_campaign=visualisation/4299524' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>"""
        st.markdown(html_string, unsafe_allow_html = True)

    # map plot
    st.subheader('Global COVID-19 Situation in 2020')
    st.plotly_chart(fig_map)

    # table
    st.subheader('Rank of Countries with the Most Cases/Deaths')
    df_rank = df_map[['location', 'date', col_name]].sort_values(col_name, ascending = False).reset_index(drop = True)
    df_rank = df_rank.reset_index().rename({'index': 'rank'}, axis = 1)
    df_rank['rank'] = df_rank['rank'] + 1
    st.write(df_rank.iloc[(num[0]-1):num[1], :])

if session == 'Economic Perspective':
    # GDP data
    GDP = pd.read_excel('./Data/G20_GDP.xlsx')
    GDP = GDP.rename({'GDP 2020 Q1': 'The First Quarter', 'GDP 2020 Q2': 'The Second Quarter'}, axis = 1)

    # region data
    region = pd.read_csv('./Data/region.csv', encoding='Windows-1252')

    # COVID and population data
    df_scatter = pd.read_csv('./Data/population and covid.csv')[['country_name', 'population', 'date', 'cases', 'deaths']]
    df_scatter = pd.merge(df_scatter[df_scatter['date'].isin(['2020-03-31', '2020-06-30'])],
                        GDP, left_on = 'country_name', right_on = 'Country')
    df_scatter = pd.merge(df_scatter, region, left_on = 'country_name', right_on = 'Country').drop(['Country_x', 'Country_y'], axis = 1)
    df_scatter['Log Cases'] = np.log(df_scatter['cases'])
    df_scatter['Log Deaths'] = np.log(df_scatter['deaths'])
    df_scatter.rename({'cases': 'Total Cases', 'deaths': 'Total Deaths'}, axis = 1, inplace = True)

    # sider
    st.sidebar.subheader("Economic Perspective")
    quarter = st.sidebar.selectbox("The Quarter to Explore on in 2020:", ['The First Quarter', 'The Second Quarter'])
    measure = st.sidebar.selectbox("Measure of Severity:", ['Log Cases', 'Log Deaths', 'Total Cases', 'Total Deaths'])
    countries = st.sidebar.multiselect('G20 Countries:', list(df_scatter['country_name'].unique()), default = list(df_scatter['country_name'].unique()))
    df_scatter = df_scatter[df_scatter['country_name'].isin(countries)][[measure] + [quarter] + ['country_name', 'population', 'Region', 'date']]
    if quarter == 'The First Quarter':
        df_scatter = df_scatter[df_scatter['date'] == '2020-03-31']
    else:
        df_scatter = df_scatter[df_scatter['date'] == '2020-06-30']
    
    # graph
    fig_bar = px.bar(df_scatter[['country_name', quarter]].sort_values(quarter, ascending = True), 'country_name', quarter)
    fig_bar.update_traces(marker_color = 'rgba(31,119,180,1)')
    fig_bar.update_xaxes(title_text = 'Country', gridcolor = 'whitesmoke', linecolor = 'black')
    fig_bar.update_layout({'plot_bgcolor': 'rgb(255,255,255)', 'paper_bgcolor': 'rgb(255,255,255)'})
    fig_bar.update_yaxes(title_text = 'GDP Growth Rate (%)', gridcolor = 'whitesmoke', linecolor = 'black')

    fig_scatter = px.scatter(df_scatter, x = measure, y = quarter, color = "Region", size = 'population', template = 'plotly_white', hover_name = "country_name", size_max = 60)
    fig_scatter.update_layout(width = 800, height = 500)
    fig_scatter.update_xaxes(title_text = measure, gridcolor = 'whitesmoke', linecolor = 'black')
    fig_scatter.update_yaxes(title_text = 'GDP Growth Rate', gridcolor = 'whitesmoke', linecolor = 'black')

    #show graph
    st.subheader('The Economic Influence of COVID-19 on G20 Countries in 2020')
    st.plotly_chart(fig_bar)

    st.subheader('The Relationship between COVID-19 Severity and Economic Influence')
    st.plotly_chart(fig_scatter)

    check_scatter = st.checkbox('Show the data')
    if check_scatter:
        st.write(df_scatter[['country_name', 'date', quarter, measure]].rename({quarter: quarter + ' GDP Change'}, axis = 1).sort_values(quarter + ' GDP Change').reset_index(drop = True))
