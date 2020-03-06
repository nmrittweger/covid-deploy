import dash
import dash_core_components as dcc
import dash_html_components as html


import pandas as pd
import plotly.express as px
from dash.dependencies import Output
import time
from datetime import datetime

app = dash.Dash()
server = app.server

#read in saved data and merge
df=pd.read_csv('https://raw.githubusercontent.com/nmrittweger/covid-19/master/data.csv')
countries = pd.read_csv('https://raw.githubusercontent.com/nmrittweger/covid-19/master/countrycodes.csv')
df=pd.merge(df, countries, how='left', on='Country')
df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['day']=df['date'].dt.day
df['yy/mm/dd']=df['year'].astype(str) + '/' + df['month'].astype(str) + '/' + df['day'].astype(str)

#lists for use in dropdowns
country_list = countries['C_Clean'].unique()
country_list=country_list.tolist()
country_list.sort()

type_list= df['type'].unique()
type_list=type_list.tolist()

date_list=df['date'].drop_duplicates()
date_list=pd.DataFrame(date_list)

region_list=df['region'].unique()
region_list=region_list.tolist()
region_list.sort()

#create region/country dictionary for dynamic dropdown menus
dfr=df[['region','Country']].drop_duplicates()
regionDict= {k: g.iloc[:,-1].values.tolist()
          for k, g in dfr.groupby('region')}
regions = list(regionDict.keys())


#create the marks for the date slider
daterange=df['date'].drop_duplicates()
def unixTimeMillis(dt):
    ''' Convert datetime to unix timestamp '''
    return int(time.mktime(dt.timetuple()))

def unixToDatetime(unix):
    ''' Convert unix timestamp to datetime. '''
    return pd.to_datetime(unix,unit='s')

def getMarks(start, end, Nth):
    ''' Returns the marks for labeling. 
        Every Nth value will be used.
    '''
    result = {}
    for i, date in enumerate(daterange):
        if(i%Nth == 1):
            # Append value to dict
            result[unixTimeMillis(date)] = str(date.strftime('%m-%d'))
    return result

marks=getMarks(daterange.min(),daterange.max(),2)



#LAYOUT**************************************************************************************************
app.layout = html.Div(dcc.Tabs(id="tabs", children=[
    dcc.Tab(label='Timeline', children=[
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label':region, 'value':region} for region in regions],
            value = list(regionDict.keys()),
            multi=True,
            style=dict(width='450px')
            ),
        dcc.Dropdown(
            id='country-dropdown',
            multi=True
            ),
        dcc.Graph(id='timeline_view'),
        dcc.Dropdown(id='timeline_type', options = [{'label':t, 'value':t} for t in type_list],value='confirmed',multi=False, style=dict(width='400px') ),
        dcc.Graph(id='timeline_country_bar'),
        ]),
    dcc.Tab(label='Global View', children=[
        dcc.Dropdown(id='g_region', options=[dict(label=i,value=i) for i in region_list ],value=region_list, multi=True, style=dict(width='450px') ),
        dcc.Dropdown(id='g_type', options=[dict(label=i,value=i) for i in type_list ],value='confirmed', style=dict(width='450px')),
        html.Div(dcc.Slider(
                id='g_date',
                min = unixTimeMillis(daterange.min()),
                max = unixTimeMillis(daterange.max()),
                value = unixTimeMillis(daterange.max()),
                step=None, #only defined marks can be selected
                marks=marks
            )),
        html.Div(id='slider-output-container'),
        html.Div(dcc.Graph(id='global_view')),
        html.Div(dcc.Graph(id='case_rank'))
        ])      
]))


#FUNCTIONS AND CALLBACKS**********************************************************************************************************

#callback to dynamically update country dropdown menu for selected regions
@app.callback(
    [dash.dependencies.Output('country-dropdown', 'options'),
     dash.dependencies.Output('country-dropdown', 'value')],
    [dash.dependencies.Input('region-dropdown', 'value')]
)
def update_country_dropdown(region):
    countrylist = []
    valuelist=[]
    for r in region:
        r_list=[{'label': i, 'value': i} for i in regionDict[r]]
        for e in r_list:
            countrylist.append(e)
    
    for r in region:
        r_list=[i for i in regionDict[r]]
        for e in r_list:
            valuelist.append(e)    
    return countrylist, valuelist


#graph cases over time for selected countries for all types
@app.callback(
    Output(component_id='timeline_view', component_property='figure'),
    [dash.dependencies.Input('country-dropdown', 'value')])
def country_view(g_country):
    #graph country over time
    dfc=df[df['Country'].isin(g_country)]
    dfc=pd.pivot_table(dfc,index=['type','date'],values='value',aggfunc='sum')
    dfc=dfc.reset_index()
    fig = px.line(dfc,x='date',y='value',color='type')
    fig.update_layout(title= 'Combined cases for selected countries')
    return fig

#graph cases over time for selected countries
@app.callback(
    Output(component_id='timeline_country_bar', component_property='figure'),
    [dash.dependencies.Input('country-dropdown', 'value'),
     dash.dependencies.Input('timeline_type', 'value')])
def timeline_country_view(g_country,g_type):
    #graph country over time
    dfc=df[df['Country'].isin(g_country)]
    dfc=dfc[dfc['type']==g_type]
    dfc=pd.pivot_table(dfc,index=['Country','date'],values='value',aggfunc='sum')
    dfc=dfc.reset_index()
    fig=px.bar(dfc,x='date',y='value',color='Country')
    return fig

#returns header for the choroplethgraph:selected date
@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('g_date', 'value')])
def update_output(value):
    g_date=datetime.utcfromtimestamp(value)
    g_date=g_date.replace(hour=0, minute=0, second=0)
    return 'As of ' + g_date.strftime('%Y-%m-%d')

#shows choroplethgraph
@app.callback(
    [Output(component_id='global_view', component_property='figure'),
     Output(component_id='case_rank', component_property='figure')],
    [dash.dependencies.Input('g_region', 'value'),
     dash.dependencies.Input('g_type', 'value'),
     dash.dependencies.Input('g_date', 'value')])
def global_view(g_region,g_type,g_date):
    g_date=datetime.utcfromtimestamp(g_date)
    g_date=g_date.replace(hour=0, minute=0, second=0)
    dfg=df[df['date']==g_date]
    dfg=dfg[dfg['type']==g_type]
    dfg=dfg[dfg['region'].isin(g_region)]
    dfg=pd.pivot_table(dfg,index=['region','City','C_Clean','Lat','Long'], values='value',aggfunc='sum')
    dfg=dfg.reset_index()
    dfg.columns=['Region','City','Country','Lat','Long',g_type]
    dfg=dfg[dfg[g_type]!=0]
    dfg['size']= dfg[g_type]/10
    dfg['size']=dfg.apply(lambda x: min(x['size'],14),axis=1)
    dfg['size']=dfg.apply(lambda x: max(x['size'],2),axis=1)    
    fig=px.scatter_geo(dfg,lat='Lat',lon='Long',color='Country',hover_name=g_type,hover_data=[g_type],size='size',projection="natural earth",width=1600,height=800,opacity=0.8,)
    dfc=pd.pivot_table(dfg, index=['Region','Country'], values=g_type, aggfunc='sum')
    dfc=dfc.reset_index()
    dfc.columns=['Region','Country',g_type]
    dfc=dfc.sort_values(by=[g_type],ascending=False)
    fig2 = px.bar(dfc, x='Country',y=g_type,color='Region')
    return fig, fig2

if __name__ == '__main__':
    app.run_server(debug=False)
