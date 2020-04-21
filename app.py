import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import math
import pandas as pd
import plotly.express as px
from dash.dependencies import Output
import time
from datetime import datetime



app = dash.Dash()
server = app.server

#read in saved data and merge
#import data from CSSE
confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')


#combine
confirmed['type']='Confirmed Cases'
recovered['type']='Recoveries'
deaths['type']='Deaths'
df= confirmed.append(recovered)
df=df.append(deaths)
df['Province/State']=df['Province/State'].fillna('N.A.')

#reformat for graphing
df= df.set_index(['Province/State','Country/Region','Lat','Long','type'])
df=df.stack()
df=df.reset_index()
df.columns=['City','Country','Lat','Long','type','date','value']
df['date']=pd.to_datetime(df['date'],infer_datetime_format=True)
df=pd.pivot_table(df, index=['City','Country','Long','Lat','date'],columns='type',values='value')
df['Currently Ill'] = df['Confirmed Cases'] - df['Deaths']-df['Recoveries']
df=df.stack()
df=df.reset_index()
df.columns=['City','Country','Long','Lat','date','type','value']

conf_date=df[df['type']=='Confirmed Cases']['date'].max()
reco_date=df[df['type']=='Recoveries']['date'].max()
deat_date=df[df['type']=='Deaths']['date'].max()
curill_date=min(conf_date,reco_date,deat_date)

#read in countries from own csv
countries = pd.read_csv('https://raw.githubusercontent.com/nmrittweger/covid-19/master/countrycodes.csv')
#merge pop into countreis
pop=pd.read_csv('https://raw.githubusercontent.com/nmrittweger/covid-19/master/population_adjusted.csv')
countries=pd.merge(countries,pop, how='left',left_on='alpha-3',right_on='Country Code')

df=pd.merge(df, countries, how='left', on='Country')
df['region']=df['region'].fillna('Locations reporting for the first time')

df['date']=pd.to_datetime(df['date'])


#testing and development below ###########################################################
#nothing to test
#############################################################################################

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

#def unixToDatetime(unix):
#    ''' Convert unix timestamp to datetime. '''
#    return pd.to_datetime(unix,unit='s')

def getMarks(start, end, Nth):
    ''' Returns the marks for labeling. 
        Every Nth value will be used.
    '''
    result = {}
    for i, date in enumerate(daterange):
        if(i%Nth == 1):
            # Append value to dict
            result[unixTimeMillis(date)] = str(date.strftime('%m-%d'))
    #append max and min date
    result[unixTimeMillis(start)] = str(start.strftime('%m-%d'))
    result[unixTimeMillis(end)] = str(end.strftime('%m-%d'))
    return result

number_of_marks=math.ceil(len(daterange)/25) #show no more than 25 date marks
marks=getMarks(daterange.min(),daterange.max(),number_of_marks)

changetypes=['Number of Cases','Cases per 1 mio inhabitants', 'Growth: Daily New Cases','Growth : New Cases per 1 mio inhabitants',
             'Growth: Daily Percentage Change','Growth: Days for Cases to Double']



#LAYOUT**************************************************************************************************
app.layout = html.Div(dcc.Tabs(id="tabs", children=[
    dcc.Tab(label='Timeline', children=[
        html.H3('Latest available information as of:'),
        html.H5(' Confirmed Cases: ' + conf_date.strftime('%Y-%m-%d') +', Recoveries: ' + reco_date.strftime('%Y-%m-%d') + ', Deaths: ' + deat_date.strftime('%Y-%m-%d') + ', Currently Ill: ' + curill_date.strftime('%Y-%m-%d')),
        html.H4('Select region or country for a timeline of confirmed, recovered and terminal cases'),
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
        dcc.Dropdown(id='timeline_type', options = [{'label':t, 'value':t} for t in type_list],value='Confirmed Cases',multi=False, style=dict(width='400px') ),
        dcc.Graph(id='timeline_country_bar'),
        dcc.Dropdown(id='change_type', options = [dict(label=i,value=i) for i in changetypes ],value=changetypes[3],multi=False, style=dict(width='400px') ),
        dcc.Dropdown(id='change_periods', options = [{'label':'Latest Number','value': 1},
                                                                        {'label':'3 Day Avg','value':3},
                                                                        {'label':'Weekly Avg','value':7},
                                                                        {'label':'14 Day Avg','value':14}],value=7,multi=False, style=dict(width='400px') ),
        dcc.Graph(id='change_over_time'),
        ]),
    dcc.Tab(label='Global View', children=[
        html.H4(''),
        html.Div(children=[
            html.Div(children=[html.Label('Regions to INCLUDE'),
                                dcc.Dropdown(id='g_region', options=[dict(label=i,value=i) for i in region_list ],value=region_list, multi=True, style=dict(width='450px') )],
                     style=dict(display='inline-block',width='50%', verticalAlign="left",padding=10)),
             html.Div(children=[html.Label('Countries to EXCLUDE'),
                                dcc.Dropdown(id='excludedCountries', options=[dict(label=i,value=i) for i in country_list ],value=[], multi=True, style=dict(width='450px') )],
                      style=dict(display='inline-block',width='45%', verticalAlign="left",padding=10)),
        html.Div(children=[
        html.Div(children=[html.Label('Metric'),dcc.Dropdown(id='changetype', options=[dict(label=i,value=i) for i in changetypes ],value=changetypes[2], style=dict(width='450px'))],
                 style=dict(display='inline-block',width='20%', verticalAlign="left",)),
        html.Div(children=[html.Label('Type of Cases'),dcc.Dropdown(id='g_type', options=[dict(label=i,value=i) for i in type_list ],value='Confirmed Cases', style=dict(width='450px'))],
                 style=dict(display='inline-block',width='20%', verticalAlign="left",)),
        html.Div(children=[html.Label('Average'),dcc.Dropdown(id='no_of_periods', options=[{'label':'Latest Number','value': 1},
                                                                        {'label':'3 Day Avg','value':3},
                                                                        {'label':'Weekly Avg','value':7},
                                                                        {'label':'14 Day Avg','value':14}],value=1, style=dict(width='150px'))],
                 style=dict(display='inline-block',width='20%', verticalAlign="left",)),
        html.Div(children=[html.Label('Show Top Countries Only'),dcc.Dropdown(id='no_of_top_countries', options=[{'label':'Show All','value': len(country_list)},
                                                                        {'label':'Top 10','value':10},
                                                                        {'label':'Top 20','value':20},
                                                                        {'label':'Top 30','value':30},
                                                                        {'label':'Top 50','value':50},],value=20, style=dict(width='450px'))],
                 style=dict(display='inline-block',width='20%', verticalAlign="left",)),
        html.Div(children=[html.Label('Only Countries with at least'),dcc.Dropdown(id='minimum_cases', options=[{'label':'Show All','value': 0},
                                                                        {'label':'100 Confirmed Cases','value':100},
                                                                        {'label':'250 Confirmed Cases','value':250},
                                                                        {'label':'500 Confirmed Cases','value':500},
                                                                        {'label':'1,000 Confirmed Cases','value':1000},
                                                                        {'label':'2,500 Confirmed Cases','value':2500},
                                                                        {'label':'5,000 Confirmed Cases','value':5000},
                                                                        {'label':'10,000 Confirmed Cases','value':10000},],value=100, style=dict(width='450px'))],
                 style=dict(display='inline-block',width='18%', verticalAlign="left",)),
        ],style=dict(display='inline-block',width='98%', verticalAlign="left",padding=10))
        ]),
        
    
    
    html.Div(dcc.Slider(
                id='g_date',
                min = unixTimeMillis(daterange.min()),
                max = unixTimeMillis(daterange.max()),
                value = unixTimeMillis(daterange.max()),
                step=None, #only defined marks can be selected
                marks=marks
            ), style= {'width': '98%', 'display': 'inline-block','marginBottom': 10, 'marginTop': 25}),
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

#graph change of case numbers over time for selected countries
@app.callback(
    Output(component_id='change_over_time', component_property='figure'),
    [dash.dependencies.Input('country-dropdown', 'value'),
     dash.dependencies.Input('timeline_type', 'value'),
     dash.dependencies.Input('change_type', 'value'),
     dash.dependencies.Input('change_periods', 'value')])
def change_over_time(g_country,g_type,change_type,change_periods):
    chng=df[df['Country'].isin(g_country)]
    chng=chng[chng['type']==g_type]
    dfc=pd.pivot_table(chng, index=['Country','date','region','Population'], values='value', aggfunc='sum')
    dfc=dfc.reset_index()
    change=pd.DataFrame()
    for c in dfc['Country'].unique().tolist():
        tdfc=dfc[dfc['Country']==c]
        tdfc['Number of Cases']= tdfc['value'].rolling(window=change_periods).mean()
        tdfc['Cases per 1 mio inhabitants']=1000000 * tdfc['Number of Cases'] / tdfc['Population']
        tdfc['Growth: Daily New Cases']= tdfc['value'].diff(periods=change_periods) / change_periods
        tdfc['Growth : New Cases per 1 mio inhabitants']=1000000 * tdfc['Growth: Daily New Cases']/ tdfc['Population']
        tdfc['Growth: Daily Percentage Change']= tdfc['value'].pct_change(periods=change_periods) / change_periods
        tdfc['Growth: Days for Cases to Double']= 1/(tdfc['value'].pct_change(periods=change_periods) / change_periods)
        tdfc=tdfc.replace([0, np.inf, -np.inf], np.nan)
        change=change.append(tdfc)

    def sorter(changetype):
        sortme=False
        catorder='total descending'
        if changetype=='Growth: Days for Cases to Double':
            sortme=True
            catorder= 'total ascending'
        return sortme, catorder

    hovrdata=changetypes.copy()
    hovrdata.append('Population')

    fig = px.line(change,x='date',y=change_type, color='Country',hover_data=hovrdata)
    if change_type=='Growth: Daily Percentage Change':
        fig.update_yaxes(range=[0,1], tickformat='%')
    
    fig.update_yaxes(categoryorder=sorter(change_type)[1])
    fig.update_layout(title=g_type + ' - ' + change_type + ' (' + str(change_periods) + ' day average)')    
    return fig


#shows choroplethgraph
@app.callback(
    [Output(component_id='global_view', component_property='figure'),
     Output(component_id='case_rank', component_property='figure')],
    [dash.dependencies.Input('g_region', 'value'),
     dash.dependencies.Input('g_type', 'value'),
     dash.dependencies.Input('g_date', 'value'),
     dash.dependencies.Input('changetype', 'value'),
     dash.dependencies.Input('no_of_periods', 'value'),
     dash.dependencies.Input('excludedCountries', 'value'),
     dash.dependencies.Input('no_of_top_countries', 'value'),
     dash.dependencies.Input('minimum_cases', 'value'),
     ])
def global_view(g_region,g_type,g_date, changetype, no_of_periods, excludedCountries,no_of_top_countries,minimum_cases):
    g_date=datetime.fromtimestamp(g_date)
    #g_date=g_date.replace(hour=0, minute=0, second=0)
    dfg=df[~df['Country'].isin(excludedCountries)]
    dfg=dfg[dfg['region'].isin(g_region)]
    dfg=dfg[dfg['type']==g_type]
    
    
    dfc=dfg.copy(deep=True) #create copy to then calculate country consolidated values for the bar chart further down
    
    #get data for choropleth seperate from country data to preserve more detailed lat and long values    
    dfg=pd.pivot_table(dfg,index=['region','City','Country','Lat','Long','date','Population'], values='value',aggfunc='sum')
    dfg=dfg.reset_index()
    dfg['Number of Cases']= dfg['value'].rolling(window=no_of_periods).mean()
    dfg['Cases per 1 mio inhabitants']=1000000 * dfg['Number of Cases'] / dfg['Population']
    dfg['Growth: Daily New Cases']= dfg['value'].diff(periods=no_of_periods) / no_of_periods
    dfg['Growth : New Cases per 1 mio inhabitants']=1000000 * dfg['Growth: Daily New Cases']/ dfg['Population']
    dfg['Growth: Daily Percentage Change']= dfg['value'].pct_change(periods=no_of_periods) / no_of_periods
    dfg['Growth: Days for Cases to Double']= 1/(dfg['value'].pct_change(periods=no_of_periods) / no_of_periods)
    dfg=dfg.replace([0,np.inf, -np.inf], np.nan)
    dfg=dfg[dfg[changetype].notnull()]
    dfg=dfg[dfg['date']==g_date]

    #consolidate by country
    dfc=pd.pivot_table(dfc, index=['Country','date','region','Population'], values='value', aggfunc='sum')
    dfc=dfc.reset_index()
    dfc['Number of Cases']= dfc['value'].rolling(window=no_of_periods).mean()
    dfc['Cases per 1 mio inhabitants']=1000000 * dfc['Number of Cases'] / dfc['Population']
    dfc['Growth: Daily New Cases']= dfc['value'].diff(periods=no_of_periods) / no_of_periods
    dfc['Growth : New Cases per 1 mio inhabitants']=1000000 * dfc['Growth: Daily New Cases']/ dfc['Population']
    dfc['Growth: Daily Percentage Change']= dfc['value'].pct_change(periods=no_of_periods) / no_of_periods
    dfc['Growth: Days for Cases to Double']= 1/(dfc['value'].pct_change(periods=no_of_periods) / no_of_periods)
    dfc=dfc.replace([0, np.inf, -np.inf], np.nan)
    dfc=dfc[dfc[changetype].notnull()]
    dfc=dfc[dfc['date']==g_date]
    
    def sorter(changetype):
        sortme=False
        catorder='total descending'
        if changetype=='Growth: Days for Cases to Double':
            sortme=True
            catorder= 'total ascending'
        return sortme, catorder
        
    dfc=dfc[dfc['Number of Cases']>minimum_cases]
    dfc=dfc.sort_values(by=[changetype],ascending=sorter(changetype)[0]).head(no_of_top_countries)
    top_countries=dfc['Country'].unique().tolist()
    
    #limit dfg to only top countries
    dfg=dfg[dfg['Country'].isin(top_countries)]
    dmin=dfg[changetype].min()
    dmax=dfg[changetype].max()
    drange=abs(dmax-dmin)
    
    def sizeme(changetype, dmin, dmax, valuetosize):
        sm = 3+(((valuetosize-dmin)/drange)*38)
        if changetype=='Growth: Days for Cases to Double':
            sm=3+(((abs(valuetosize-dmax))/drange)*38)
        return sm
    dfg
    dfg['size']=dfg.apply(lambda x: sizeme(changetype,dmin,dmax,x[changetype]),axis=1)

    hovrdata=changetypes.copy()
    hovrdata.append('Population')
    
    #Graph it
    fig=px.scatter_geo(dfg,lat='Lat',lon='Long',color='Country',hover_name=dfg['Country'],hover_data=hovrdata,
                       size='size',projection="natural earth",width=1600,height=800,opacity=0.8,)    
    title='As of ' + g_date.strftime('%Y-%m-%d')    
    fig.update_layout(title=title)

    fig2 = px.bar(dfc, x='Country',y=changetype,color='region', hover_data=hovrdata)
    fig2.update_xaxes(tickangle=45,categoryorder=sorter(changetype)[1])
    
    return fig, fig2



if __name__ == '__main__':
    app.run_server(debug=False)
