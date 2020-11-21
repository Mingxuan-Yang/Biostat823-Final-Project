# streamlit run covid-19.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta
# import altair as alt
import plotly.express as px
import plotly.graph_objs as go
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

session = st.sidebar.selectbox("Section", ["Welcome!", "Overview", "Economic Perspective", "Dietary Perspective"])
st.title('COVID-19 Dashboard')

if session == "Welcome!":

    st.sidebar.subheader("Welcome to our dashboard!")

    # image
    image = Image.open('img/covid-19.png')
    st.image(image, width = 700)
    st.subheader("Introduction")
    st.write("""
    This is a dashboard about COVID-19, its economic influence and the potential dietary method to prevent it. 
    The target audience of this dashboard is the general audience. The major objective is to help the audience 
    gain a general understanding about COVID-19. It also serves as a tool to exhibit the explorary data analysis 
    part as well as the results of our machine learning algorithms for the final project in course BIOSTAT 823 
    (Statistical Program for Big Data) at Duke University.  
    This dashboard is divided into the following parts:
    - Welcome!
    - Overview
        - Worldwide Overview
        - Bar Chart Race
        - Individual Country
    - Economic Perspective
        - G20
        - G7
    - Dietary Perspective
        - EDA
        - Model
    """)
    st.subheader("Data Source")
    st.write("""
    - **COVID-19 Datasets**  
    The COVID-19 datasets are from [Amazon Web Services (AWS) data lake](https://dj2taa9i652rf.cloudfront.net/). 
    This database is a centralized repository of up-to-date and curated datasets about COVID-19.

    - **Food Datasets**  
    The diet information and corresponding COVID-19 recovery rate data are from [Kaggle](https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset?select=Fat_Supply_Quantity_Data.csv).

    - **Economic Datasets**  
    The economic impacts of COVID-19, represented by GDP growth rate of G20 countries, are provided by Organisation for Economic Co-operation and Development (OECD).""")

if session == "Overview":
    st.sidebar.subheader("Overview")
    parts = st.sidebar.radio("Three Parts:", ["Worldwide Overview", "Bar Chart Race", "Individual Country"])
    if parts == 'Worldwide Overview':

        df_line = pd.read_csv('./Data/by_country.csv')[['location', 'date', 'total_cases', 'total_deaths']]
        df_line = df_line[df_line['location'] == 'World']
        date = df_line['date']
        df_line = df_line.rolling(3).mean()
        df_line['date'] = pd.to_datetime(date).dt.date
        df_line = df_line.dropna()
        df_line.rename({'total_cases': 'Total Cases', 'total_deaths': 'Total Deaths'}, axis = 1, inplace = True)

        df_map = pd.read_csv('./Data/by_country.csv')
        df_map['date'] = pd.to_datetime(df_map['date']).dt.date
        df_map = df_map[~df_map['location'].isin(['World', 'International'])]

        #sidebar
        # date_range = st.sidebar.date_input('Range of date:', [min(df_line['date']), max(df_line['date'])],min_value = min(df_line['date']), max_value = max(df_line['date']))
        st.sidebar.subheader('Parameters for the Line Plot')
        date_range =  st.sidebar.slider('Range of date:', min(df_line['date']), max(df_line['date']), (pd.to_datetime('2020-03-01').date(), pd.to_datetime('2020-10-01').date()))
        cols = st.sidebar.multiselect('Measure of Severity:', df_line.columns[:2].to_list(), default = df_line.columns[:2].to_list())
        measure = st.sidebar.selectbox("Plot Every:", ['1 day', '5 days', '7 days', '30 days'], index = 2)
        st.sidebar.subheader('Parameters for the Map Plot')
        dates1 = st.sidebar.date_input('Date:', datetime.date(2020, 11, 1), min_value=min(df_map['date']),
                                      max_value=max(df_map['date']))
        method1 = st.sidebar.selectbox("Measure of Severity:", ['Cases', 'Deaths'])
        method2 = st.sidebar.selectbox("Calculation Method:", ['Total', 'New', 'Total per Million', 'New per Million'])

        measure = int(measure[:2])
        dates = [date_range[0]]
        while dates[-1] <= date_range[1]:
            dates.append(dates[-1] + timedelta(days = 1))
        df_line = df_line[df_line['date'].isin(dates)][cols + ['date']]

        #show graph
        fig_line = go.Figure()
        colors = ['rgba(255,127,14,1)', 'rgba(31,119,180,1)']

        for i in cols:
            df_line[i] = round(df_line[i], 0)
            fig_line.add_trace(go.Scatter(x = df_line['date'][::measure],
                                          y = df_line[i][::measure],
                                          marker = dict(color = colors[i == 'Total Cases']),
                                          mode = 'lines+markers',
                                          name = i))
        fig_line.update_xaxes(title_text = 'Date', gridcolor = 'whitesmoke', linecolor = 'black')
        fig_line.update_layout({'plot_bgcolor': 'rgb(255,255,255)', 'paper_bgcolor': 'rgb(255,255,255)'})
        fig_line.update_yaxes(title_text = 'Total Number', gridcolor = 'whitesmoke', linecolor = 'black')

        # map graph
        df_map = df_map[df_map['date'] == dates1]
        if len(method2.split()) > 1:
            col_name = method2.lower().split()[0] + '_' + method1.lower() + '_' + '_'.join(method2.lower().split()[1:])
        else:
            col_name = method2.lower() + '_' + method1.lower()
        fig_map = go.Figure(data=go.Choropleth(
            locations=df_map['location'],
            z=df_map[col_name].astype(float),
            locationmode='country names',
            colorscale='Reds',
            autocolorscale=False,
            marker_line_color='black',
            # text = '<b>' + dff['State'] + '</b><br>' + field + ' Field: ' + dff[field].astype(int).astype(str)
        ))

        fig_map.update_layout(
            width=700, height=500,
            geo=dict(showlakes=True, lakecolor='rgb(255, 255, 255)'))

        # plot
        st.write("""The **Worldwide Overview** part provides two figures about COVID-19. The first figure shows the total cases 
        and deaths around the world. This indicates the overall trend of COVID-19 for different seasons and the whole severity. 
        The second figure is the geographical distribution of COVID-19, represented by different measure of severity and 
        calculation method. This offers a reference about which region around the world should the World Health Organization 
        pays more attention to.""")
        st.subheader('Total Number of COVID-19 Cases/Deaths by Date')
        st.write("""This is an overview figure from January 10, 2020 to November 6, 2020. Certain date range can be selected using the sidebar.""")
        st.plotly_chart(fig_line)

        with st.beta_expander("Figure Details"):
             st.write("""
                The data is from [Amazon Web Services (AWS) data lake](https://aws.amazon.com/blogs/big-data/a-public-data-lake-for-analysis-of-covid-19-data/). The cases and deaths for each day is calculated by the rolling mean with a window of 3 days.
             """)

        st.subheader('Global COVID-19 Situation in 2020')
        st.write(
            """This is the geographical distribution of COVID-19. The gray part means that the corresponding data is missing.""")
        st.plotly_chart(fig_map)

    if parts == 'Bar Chart Race':
        df_map = pd.read_csv('./Data/by_country.csv')
        df_map['date'] = pd.to_datetime(df_map['date']).dt.date
        df_map = df_map[~df_map['location'].isin(['World', 'International'])]

        check_map = st.sidebar.checkbox('Show Animation Plot', value=True)
        dates = st.sidebar.date_input('Date:', datetime.date(2020, 11, 1), min_value=min(df_map['date']),
                                      max_value=max(df_map['date']))
        method1 = st.sidebar.selectbox("Measure of Severity:", ['Cases', 'Deaths'])
        method2 = st.sidebar.selectbox("Calculation Method:", ['Total', 'New', 'Total per Million', 'New per Million'])
        num = st.sidebar.slider('Number of Top Countries:', 1, 100, (1, 5))
        if len(method2.split()) > 1:
            col_name = method2.lower().split()[0] + '_' + method1.lower() + '_' + '_'.join(method2.lower().split()[1:])
        else:
            col_name = method2.lower() + '_' + method1.lower()

        st.sidebar.write('The default arrangement is from highest to lowest.')
        change_arrange = st.sidebar.checkbox("From Lowest to Highest")

        # bar chart race
        st.write("""The **Bar Chart Race** section provides a bar chart race and a table about COVID-19. 
        The bar chart race shows the transition of epidemic center since its outbreak. The final result is 
        that from Feburary 1, 2020 to November 7, 2020, the epidemic center has been moved from China, the origin 
        of COVID-19, to USA, the country with most total cases and deaths according to **Woldwide Overview** section. 
        The table exhibits the detailed ranking of countries using different measure of severity and 
        calculation method for each date. We hope this dashboard will be helpful for the audience's perception about 
        COVID-19 distribution.""")
        if check_map:
            st.subheader('COVID-19 Cases from February to November at Country Level')
            html_string = """<iframe src='https://flo.uri.sh/visualisation/4299524/embed' title='Interactive or visual content' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/4299524/?utm_source=embed&utm_campaign=visualisation/4299524' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>"""
            st.write("""This is a bar chart race of COVID-19 cases from February 1, 2020 to November 7, 2020. For time saving purpose, the figure is shown every 5 days.""")
            st.markdown(html_string, unsafe_allow_html = True)

        # table
        st.subheader('Rank of Countries with the Most Cases/Deaths')
        if change_arrange:
            df_rank = df_map[['location', 'date', col_name]].sort_values(col_name, ascending=True).reset_index(drop=True)
        else:
            df_rank = df_map[['location', 'date', col_name]].sort_values(col_name, ascending=False).reset_index(drop=True)
        df_rank = df_rank.reset_index().rename({'index': 'rank'}, axis=1)
        df_rank['rank'] = df_rank['rank'] + 1
        st.write(df_rank.iloc[(num[0] - 1):num[1], :])


    if parts == "Individual Country":
        df_line = pd.read_csv('Data/by_country.csv')[['location', 'date', 'total_cases', 'total_deaths']]
        df_line = df_line[~df_line['location'].isin(['World', 'International'])]

        # sidebar
        con = st.sidebar.selectbox("Select a Country:", list(np.unique(df_line['location'])))
        df_line = df_line[df_line['location'] == con]
        df_line['date'] = pd.to_datetime(df_line['date']).dt.date
        df_line = df_line.dropna()
        df_line.rename({'total_cases': 'Total Cases', 'total_deaths': 'Total Deaths'}, axis=1, inplace=True)
        date_range = st.sidebar.slider('Range of date:', min(df_line['date']), max(df_line['date']),
                                       (pd.to_datetime('2020-05-01').date(), pd.to_datetime('2020-09-01').date()))
        cols = st.sidebar.multiselect('Measure of Severity:', ['Total Cases', 'Total Deaths'],
                                      default=['Total Cases', 'Total Deaths'])
        measure = st.sidebar.selectbox("Plot Every:", ['1 day', '5 days', '7 days', '30 days'], index=2)
        measure = int(measure[:2])
        date_table = st.sidebar.date_input('Table Date:', max(df_line['date']), min_value=min(df_line['date']),
                                           max_value=max(df_line['date']))

        df_table = pd.read_csv('./Data/by_country.csv')
        df_table.drop('iso_code', axis = 1, inplace = True)
        df_table['date'] = pd.to_datetime(df_table['date']).dt.date
        df_table = df_table[(df_table['location'] == con) & (df_table['date'] == date_table)]

        fig_line1 = go.Figure()
        colors = ['rgba(255,127,14,1)', 'rgba(31,119,180,1)']

        for i in cols:
            df_line[i] = round(df_line[i], 0)
            fig_line1.add_trace(go.Scatter(x=df_line['date'][::measure],
                                          y=df_line[i][::measure],
                                          marker=dict(color=colors[i == 'Total Cases']),
                                          mode='lines+markers',
                                          name=i))
        fig_line1.update_xaxes(title_text='Date', gridcolor='whitesmoke', linecolor='black')
        fig_line1.update_layout({'plot_bgcolor': 'rgb(255,255,255)', 'paper_bgcolor': 'rgb(255,255,255)'})
        fig_line1.update_yaxes(title_text='Total Number', gridcolor='whitesmoke', linecolor='black')

        st.write("""The **Individual Country** section provides the COVID-19 situation for each individual country 
        around the world. The figure part contains total cases and total deaths of COVID-19. And the table part 
        includes many other measurement method.""")
        st.subheader(f'Total Number of COVID-19 Cases/Deaths For {con}')
        st.plotly_chart(fig_line1)
        st.subheader(f'Situation for {con} on {date_table}')
        st.write(df_table)

if session == 'Economic Perspective':
    st.sidebar.subheader("Economic Perspective")
    parts = st.sidebar.radio("Select a Country Group:", ["G20", "G7"])

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

    # sidebar
    quarter = st.sidebar.selectbox("The Quarter to Explore on in 2020:", ['The First Quarter', 'The Second Quarter'])
    measure = st.sidebar.selectbox("Measure of Severity:", ['Log Cases', 'Log Deaths', 'Total Cases', 'Total Deaths'])
    if quarter == 'The First Quarter':
        df_scatter = df_scatter[df_scatter['date'] == '2020-03-31']
    else:
        df_scatter = df_scatter[df_scatter['date'] == '2020-06-30']

    g7_cons = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom', 'United States']

    if parts == 'G20':
        countries = st.sidebar.multiselect('G20 Countries:', list(df_scatter['country_name'].unique()), default=list(df_scatter['country_name'].unique()))
        df_scatter = df_scatter[df_scatter['country_name'].isin(countries)][[measure] + [quarter] + ['country_name', 'population', 'Region', 'date']]
    if parts == "G7":
        countries = st.sidebar.multiselect('G7 Countries:', g7_cons, default = g7_cons)

    df_scatter = df_scatter[df_scatter['country_name'].isin(countries)][[measure] + [quarter] + ['country_name', 'population', 'Region', 'date']]
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
    st.write("""This section roughly reveals the economic influence of COVID-19, denoted by the GDP growth 
    rate since outbreak at country level. Since economic perspective is not our focus, this part only serves
    as a general tool to extend the audience's perception about COVID-19.  
    The Group of Twenty (G20) is an international forum for the governments and central bank governors from 19 countries 
    and the European Union (EU). Its members represent 85 percent of global economic output. The Group of Seven (G7) is 
    an intergovernmental organization consisting of 7 countries. It is mainly about political issues and represents 58% 
    of the global net wealth. For the study of economic field, these groups are great representatives.""")
    st.subheader('The Economic Influence of COVID-19 on G20 Countries in 2020')
    st.plotly_chart(fig_bar)

    st.subheader('The Relationship between COVID-19 Severity and Economic Influence')
    st.plotly_chart(fig_scatter)

    check_scatter = st.checkbox('Show the data')
    if check_scatter:
        df_table = df_scatter[['country_name', 'date', quarter, measure]].rename({quarter: quarter + ' GDP Change', 'country_name': 'Country Name', 'date': 'Date'},
                                                                      axis=1)
        order_by = st.selectbox("Order By:", df_table.columns[[0,2,3]])
        arr_by = st.selectbox("Arrangement:", ['Ascending', 'Descending'])
        st.write(df_table.sort_values(order_by, ascending = arr_by == 'Ascending').reset_index(drop=True))

if session == "Dietary Perspective":

    region = pd.read_csv('Data/region.csv', encoding='Windows-1252')
    df_inter = pd.read_csv('Data/clean_data_countries.csv', index_col=0)
    df_inter = pd.merge(df_inter, region, on='Country')

    df_importance = pd.read_csv('Data/importance.csv')
    st.write("""This part shows the exploratory data analysis process as well as the model results 
    of the machine learning part for the final project in course BIOSTAT 823 (Statistical Program 
    for Big Data) at Duke University. The major objective of the algorithms is to explore the 
    relationship between the recovery rate of COVID-19 and the food products.""")
    # sidebar
    st.sidebar.subheader("Dietary Perspective")
    parts = st.sidebar.radio("Two Parts:", ["EDA", "Model"])
    if parts == 'EDA':
        st.sidebar.write("""Outcome: COVID-19 Recovery Rate  
        Predictors: Food Products""")
        selection = st.sidebar.selectbox("Comparison Between:", ["Predictors vs Outcome", "Predictors vs Predictors"])
        if selection == 'Predictors vs Outcome':
            cat = st.sidebar.selectbox("Food Product:", list(df_inter.columns[1:-2]))
            check_logx = st.sidebar.checkbox('Log x Axis', value=False)
            check_logy = st.sidebar.checkbox('Log y Axis', value=False)
            continents = st.sidebar.multiselect("Continent:", list(df_inter['Region'].unique()), default=list(df_inter['Region'].unique()))

            # figure
            df_inter = df_inter.loc[df_inter['Region'].isin(continents), [cat, 'Recover %', 'Region']]
            if check_logx:
                df_inter[cat] = np.log(df_inter[cat] + 3)
            if check_logy:
                df_inter['Recover %'] = np.log(df_inter['Recover %'] + 1)
            fig1 = px.scatter(df_inter, x = cat, y = 'Recover %', color = 'Region',
                              template = 'plotly_white')
            fig1.update_layout(width=700, height=500)
            name_check = ['', 'Log ']
            fig1.update_xaxes(title_text=name_check[int(check_logx)] + cat, gridcolor='whitesmoke', linecolor='black')
            fig1.update_yaxes(title_text=name_check[int(check_logy)] + 'COVID-19 Recovery Rate', gridcolor='whitesmoke', linecolor='black')

            st.subheader(f'The Relationship between {cat.title()} and COVID-19 Recovery Rate')
            st.plotly_chart(fig1)

            with st.beta_expander("Figure Details"):
                st.write("""For the logarithmic form, to obtain valid transformation. The x axis and y
                axis are calculated by  
                new x = log(x + 3)  
                new y = log(y + 1)  
                The additional value is based on the minimum value of overall variables.""")

        else:
            cat1 = st.sidebar.selectbox("Food Product for x Axis:", list(df_inter.columns[1:-2]))
            cat2 = st.sidebar.selectbox("Food Product for y Axis:", list(df_inter.columns[1:-2]))
            check_logx = st.sidebar.checkbox('Log x Axis', value=False)
            check_logy = st.sidebar.checkbox('Log y Axis', value=False)
            continents = st.sidebar.multiselect("Continent:", list(df_inter['Region'].unique()), default=list(df_inter['Region'].unique()))

            # figure
            df_inter = df_inter.loc[df_inter['Region'].isin(continents), [cat1, cat2, 'Region']]
            if check_logx:
                df_inter[cat1] = np.log(df_inter[cat1] + 3)
            if check_logy:
                df_inter[cat2] = np.log(df_inter[cat2] + 3)

            if cat1 == cat2:
                df_inter = df_inter.iloc[:, [0, 2]]

            fig1 = px.scatter(df_inter, x = cat1, y = cat2, color = 'Region',
                                  template = 'plotly_white')
            fig1.update_layout(width=700, height=500)
            name_check = ['', 'Log ']
            fig1.update_xaxes(title_text=name_check[int(check_logx)] + cat1, gridcolor='whitesmoke', linecolor='black')
            fig1.update_yaxes(title_text=name_check[int(check_logy)] + cat2, gridcolor='whitesmoke', linecolor='black')

            st.subheader(f'The Relationship between {cat1.title()} and {cat2.title()}')
            st.plotly_chart(fig1)

            with st.beta_expander("Figure Details"):
                st.write("""For the logarithmic form, to obtain valid transformation. The x axis and y
                axis are calculated by  
                new x = log(x + 3)  
                new y = log(y + 3)  
                The additional value is based on the minimum value of overall variables.""")

            st.subheader(f'Correlation matrix for food products')

            st.image('img/corr_matrix.png')

    if parts == "Model":
        selection = st.sidebar.selectbox("Select a Model:", ["Random Forest Regression", "Elastic Net Regression", "Gradient Boosting Regression"])
        num_var = st.sidebar.number_input("Number of Variables to Show (Up to 23):", 1, 23, 23)
        st.sidebar.write('The default arrangement is from most important to least important feature.')
        change_arrange = st.sidebar.checkbox("From Least Important to Most Important Feature")
        df_import = pd.read_csv('Data/importance.csv')
        models = "Random Forest Regression", "Elastic Net Regression", "Gradient Boosting Regression"
        wall_times = [651, 6.7, 212]
        mses = [0.04869, 0.04996, 0.05167]

        if change_arrange:
            fig2 = px.bar(df_import.sort_values(selection, ascending=True).iloc[:num_var, :], 'index', selection)
        else:
            fig2 = px.bar(df_import.sort_values(selection, ascending=False).iloc[:num_var, :], 'index', selection)
        fig2.update_traces(marker_color='rgba(31,119,180,1)')
        fig2.update_xaxes(title_text='Feature', gridcolor='whitesmoke', linecolor='black')
        fig2.update_layout({'plot_bgcolor': 'rgb(255,255,255)', 'paper_bgcolor': 'rgb(255,255,255)'})
        fig2.update_yaxes(title_text='Importance', gridcolor='whitesmoke', linecolor='black')

        st.subheader(f'Feature Importance Plot for {selection}')
        st.plotly_chart(fig2)
        st.subheader(f'Prediction Result of {selection}')
        st.write(f'The MSE result: {mses[models.index(selection)]}')
        st.write(f'The Wall Time of Grid Search CV: {wall_times[models.index(selection)]}s')
