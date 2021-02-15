import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))



df_nse = pd.read_csv("BPCL.csv")

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']


data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:2920,:]
valid=dataset[2920:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model("D:\\Data Science\\Projects\\Stock Martket\\saved_lstm_model.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:2920]
valid=new_data[2920:]
valid['Predictions']=closing_price



df = pd.read_csv("nifty_50_all.csv")

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

stocks_list=df.Symbol.unique().tolist()
d={}

for index, value in enumerate(stocks_list):
    d[index] = value

stocks = dict([(value, key) for key, value in d.items()])

l= ['Adani Ports','Asian Paints','Axis Bank', 'Bajaj-Auto','Bajaj Finserv','Bajaj Auto Finance',
    'Bajaj Finance','Bharti Airtel','BPCL', 'Britannia','Cipla','Coal India',
    'Dr. Reddy','Eicher Motors','GAIL', 'Grasim','HCL Technologies','Housing Development Finance Corporation Ltd.',
    'HDFC Bank Ltd.','Hero Motocorp','Hindalco', 'Hindustan Unilever','ICICI Bank','Indusind Bank',
    'Infosys Technologies Ltd.','Bharati Infratel','IOC Ltd.', 'ITC Ltd.','JSW Steels Ltd','Kotak Mahindra Bank Ltd',
    'Larsen & Toubro Ltd.','Mahindra & Mahindra Ltd.','Maruti Suzuki India Ltd.', 'MUNDRAPORT','Nestle India Ltd','NTPC Ltd.',
    'ONGC Ltd.','PowerGrid Corporation of India Ltd.','Reliance Industries Ltd.', 'State Bank of India','Sesagoa','Shree Cement Ltd.',
    'SSLT','Sun Pharmaceutical Industries Ltd.','Tata Motors Ltd.', 'Tata Steel Ltd.','Tata Consultancy Services Ltd.','Tech Mahindra Ltd.',
    'Telco','TISCO','Titan Company Ltd.', 'UltraTech Cement Ltd.','Uniphos','UPL Ltd.',
    'Vedanta Ltd.','Wipro Ltd.','Zee Entertainment Enterprises Ltd.', 'Zee Tele',
    ]

l_ordered = list(stocks.keys())
l_ordered.sort()

for index in range(len(l_ordered)):
    #print(index)
    stocks[l_ordered[index]] = l[index]

app.layout = html.Div(style={'backgroundColor': '#8ee4af'}, children = [
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='BPCL Stock Data 2000-20',children=[
			html.Div([
				html.H2("Actual Closing Price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=train.index,
								y=valid["Close"],
								mode='lines'
							)

						],
						"layout":go.Layout(
							title='Line Plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("Actual Closing Price Data",style={"textAlign": "center",'color': colors['text']}),
				dcc.Graph(
					id="Actual Closing Price Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["Close"],
								mode='lines'
							)

						],
						"layout":go.Layout(
                            title='Line Plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}
				),
				html.H2("LSTM Predicted Closing Price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["Predictions"],
								mode='lines'
							)

						],
						"layout":go.Layout(
							title='Line Plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}
				)
			])        		


        ]),
         dcc.Tab(label='Nifty Stock Data over 2000-2020', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{ "label"  :"MUNDRAPORT"  ,  "value"  :"MUNDRAPORT" } ,
                                      { "label"  :"Adani Ports"  ,  "value"  :"ADANIPORTS" } ,
                                      { "label"  :"Asian Paints"  ,  "value"  :"ASIANPAINT" } ,
                                      { "label"  :"Axis Bank"  ,  "value"  :"AXISBANK" } ,
                                      { "label"  :"Bajaj-Auto"  ,  "value"  :"BAJAJ-AUTO" } ,
                                      { "label"  :"Bajaj Finserv"  ,  "value"  :"BAJAJFINSV" } ,
                                      { "label"  :"Bajaj Auto Finance"  ,  "value"  :"BAJAUTOFIN" } ,
                                      { "label"  :"Bajaj Finance"  ,  "value"  :"BAJFINANCE" } ,
                                      { "label"  :"Bharti Airtel"  ,  "value"  :"BHARTIARTL" } ,
                                      { "label"  :"BPCL"  ,  "value"  :"BPCL" } ,
                                      { "label"  :"Britannia"  ,  "value"  :"BRITANNIA" } ,
                                      { "label"  :"Cipla"  ,  "value"  :"CIPLA" } ,
                                      { "label"  :"Coal India"  ,  "value"  :"COALINDIA" } ,
                                      { "label"  :"Dr. Reddy"  ,  "value"  :"DRREDDY" } ,
                                      { "label"  :"Eicher Motors"  ,  "value"  :"EICHERMOT" } ,
                                      { "label"  :"GAIL"  ,  "value"  :"GAIL" } ,
                                      { "label"  :"Grasim"  ,  "value"  :"GRASIM" } ,
                                      { "label"  :"HCL Technologies"  ,  "value"  :"HCLTECH" } ,
                                      { "label"  :"Housing Development Finance Corporation Ltd."  ,  "value"  :"HDFC" } ,
                                      { "label"  :"HDFC Bank Ltd."  ,  "value"  :"HDFCBANK" } ,
                                      { "label"  :"Hero Motocorp"  ,  "value"  :"HEROMOTOCO" } ,
                                      { "label"  :"Hindalco"  ,  "value"  :"HINDALCO" } ,
                                      { "label"  :"Hindustan Unilever"  ,  "value"  :"HINDUNILVR" } ,
                                      { "label"  :"ICICI Bank"  ,  "value"  :"ICICIBANK" } ,
                                      { "label"  :"Indusind Bank"  ,  "value"  :"INDUSINDBK" } ,
                                      { "label"  :"Bharati Infratel"  ,  "value"  :"INFRATEL" } ,
                                      { "label"  :"Infosys Technologies Ltd."  ,  "value"  :"INFOSYSTCH" } ,
                                      { "label"  :"IOC Ltd."  ,  "value"  :"IOC" } ,
                                      { "label"  :"ITC Ltd."  ,  "value"  :"ITC" } ,
                                      { "label"  :"JSW Steels Ltd"  ,  "value"  :"JSWSTEEL" } ,
                                      { "label"  :"Kotak Mahindra Bank Ltd"  ,  "value"  :"KOTAKBANK" } ,
                                      { "label"  :"Larsen & Toubro Ltd."  ,  "value"  :"LT" } ,
                                      { "label"  :"Mahindra & Mahindra Ltd."  ,  "value"  :"M&M" } ,
                                      { "label"  :"Maruti Suzuki India Ltd."  ,  "value"  :"MARUTI" } ,
                                      { "label"  :"Nestle India Ltd"  ,  "value"  :"NESTLEIND" } ,
                                      { "label"  :"NTPC Ltd."  ,  "value"  :"NTPC" } ,
                                      { "label"  :"ONGC Ltd."  ,  "value"  :"ONGC" } ,
                                      { "label"  :"PowerGrid Corporation of India Ltd."  ,  "value"  :"POWERGRID" } ,
                                      { "label"  :"Reliance Industries Ltd."  ,  "value"  :"RELIANCE" } ,
                                      { "label"  :"State Bank of India"  ,  "value"  :"SBIN" } ,
                                      { "label"  :"Shree Cement Ltd."  ,  "value"  :"SHREECEM" } ,
                                      { "label"  :"Sun Pharmaceutical Industries Ltd."  ,  "value"  :"SUNPHARMA" } ,
                                      { "label"  :"Telco"  ,  "value"  :"TELCO" } ,
                                      { "label"  :"Tata Motors Ltd."  ,  "value"  :"TATAMOTORS" } ,
                                      { "label"  :"TISCO"  ,  "value"  :"TISCO" } ,
                                      { "label"  :"Tata Steel Ltd."  ,  "value"  :"TATASTEEL" } ,
                                      { "label"  :"Tata Consultancy Services Ltd."  ,  "value"  :"TCS" } ,
                                      { "label"  :"Tech Mahindra Ltd."  ,  "value"  :"TECHM" } ,
                                      { "label"  :"Titan Company Ltd."  ,  "value"  :"TITAN" } ,
                                      { "label"  :"UltraTech Cement Ltd."  ,  "value"  :"ULTRACEMCO" } ,
                                      { "label"  :"Uniphos"  ,  "value"  :"UNIPHOS" } ,
                                      { "label"  :"UPL Ltd."  ,  "value"  :"UPL" } ,
                                      { "label"  :"Sesagoa"  ,  "value"  :"SESAGOA" } ,
                                      { "label"  :"SSLT"  ,  "value"  :"SSLT" } ,
                                      { "label"  :"Vedanta Ltd."  ,  "value"  :"VEDL" } ,
                                      { "label"  :"Wipro Ltd."  ,  "value"  :"WIPRO" } ,
                                      { "label"  :"Zee Tele"  ,  "value"  :"ZEETELE" } ,
                                      { "label"  :"Zee Entertainment Enterprises Ltd."  ,  "value"  :"ZEEL" } ,
                                      ], 
                             multi=True,value=['BPCL'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "56%", 'backgroundColor': '#d1e8e2',
                                    'font': {'textcolor': 'black'}}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{ "label"  :"MUNDRAPORT"  ,  "value"  :"MUNDRAPORT" } ,
                                      { "label"  :"Adani Ports"  ,  "value"  :"ADANIPORTS" } ,
                                      { "label"  :"Asian Paints"  ,  "value"  :"ASIANPAINT" } ,
                                      { "label"  :"Axis Bank"  ,  "value"  :"AXISBANK" } ,
                                      { "label"  :"Bajaj-Auto"  ,  "value"  :"BAJAJ-AUTO" } ,
                                      { "label"  :"Bajaj Finserv"  ,  "value"  :"BAJAJFINSV" } ,
                                      { "label"  :"Bajaj Auto Finance"  ,  "value"  :"BAJAUTOFIN" } ,
                                      { "label"  :"Bajaj Finance"  ,  "value"  :"BAJFINANCE" } ,
                                      { "label"  :"Bharti Airtel"  ,  "value"  :"BHARTIARTL" } ,
                                      { "label"  :"BPCL"  ,  "value"  :"BPCL" } ,
                                      { "label"  :"Britannia"  ,  "value"  :"BRITANNIA" } ,
                                      { "label"  :"Cipla"  ,  "value"  :"CIPLA" } ,
                                      { "label"  :"Coal India"  ,  "value"  :"COALINDIA" } ,
                                      { "label"  :"Dr. Reddy"  ,  "value"  :"DRREDDY" } ,
                                      { "label"  :"Eicher Motors"  ,  "value"  :"EICHERMOT" } ,
                                      { "label"  :"GAIL"  ,  "value"  :"GAIL" } ,
                                      { "label"  :"Grasim"  ,  "value"  :"GRASIM" } ,
                                      { "label"  :"HCL Technologies"  ,  "value"  :"HCLTECH" } ,
                                      { "label"  :"Housing Development Finance Corporation Ltd."  ,  "value"  :"HDFC" } ,
                                      { "label"  :"HDFC Bank Ltd."  ,  "value"  :"HDFCBANK" } ,
                                      { "label"  :"Hero Motocorp"  ,  "value"  :"HEROMOTOCO" } ,
                                      { "label"  :"Hindalco"  ,  "value"  :"HINDALCO" } ,
                                      { "label"  :"Hindustan Unilever"  ,  "value"  :"HINDUNILVR" } ,
                                      { "label"  :"ICICI Bank"  ,  "value"  :"ICICIBANK" } ,
                                      { "label"  :"Indusind Bank"  ,  "value"  :"INDUSINDBK" } ,
                                      { "label"  :"Bharati Infratel"  ,  "value"  :"INFRATEL" } ,
                                      { "label"  :"Infosys Technologies Ltd."  ,  "value"  :"INFOSYSTCH" } ,
                                      { "label"  :"IOC Ltd."  ,  "value"  :"IOC" } ,
                                      { "label"  :"ITC Ltd."  ,  "value"  :"ITC" } ,
                                      { "label"  :"JSW Steels Ltd"  ,  "value"  :"JSWSTEEL" } ,
                                      { "label"  :"Kotak Mahindra Bank Ltd"  ,  "value"  :"KOTAKBANK" } ,
                                      { "label"  :"Larsen & Toubro Ltd."  ,  "value"  :"LT" } ,
                                      { "label"  :"Mahindra & Mahindra Ltd."  ,  "value"  :"M&M" } ,
                                      { "label"  :"Maruti Suzuki India Ltd."  ,  "value"  :"MARUTI" } ,
                                      { "label"  :"Nestle India Ltd"  ,  "value"  :"NESTLEIND" } ,
                                      { "label"  :"NTPC Ltd."  ,  "value"  :"NTPC" } ,
                                      { "label"  :"ONGC Ltd."  ,  "value"  :"ONGC" } ,
                                      { "label"  :"PowerGrid Corporation of India Ltd."  ,  "value"  :"POWERGRID" } ,
                                      { "label"  :"Reliance Industries Ltd."  ,  "value"  :"RELIANCE" } ,
                                      { "label"  :"State Bank of India"  ,  "value"  :"SBIN" } ,
                                      { "label"  :"Shree Cement Ltd."  ,  "value"  :"SHREECEM" } ,
                                      { "label"  :"Sun Pharmaceutical Industries Ltd."  ,  "value"  :"SUNPHARMA" } ,
                                      { "label"  :"Telco"  ,  "value"  :"TELCO" } ,
                                      { "label"  :"Tata Motors Ltd."  ,  "value"  :"TATAMOTORS" } ,
                                      { "label"  :"TISCO"  ,  "value"  :"TISCO" } ,
                                      { "label"  :"Tata Steel Ltd."  ,  "value"  :"TATASTEEL" } ,
                                      { "label"  :"Tata Consultancy Services Ltd."  ,  "value"  :"TCS" } ,
                                      { "label"  :"Tech Mahindra Ltd."  ,  "value"  :"TECHM" } ,
                                      { "label"  :"Titan Company Ltd."  ,  "value"  :"TITAN" } ,
                                      { "label"  :"UltraTech Cement Ltd."  ,  "value"  :"ULTRACEMCO" } ,
                                      { "label"  :"Uniphos"  ,  "value"  :"UNIPHOS" } ,
                                      { "label"  :"UPL Ltd."  ,  "value"  :"UPL" } ,
                                      { "label"  :"Sesagoa"  ,  "value"  :"SESAGOA" } ,
                                      { "label"  :"SSLT"  ,  "value"  :"SSLT" } ,
                                      { "label"  :"Vedanta Ltd."  ,  "value"  :"VEDL" } ,
                                      { "label"  :"Wipro Ltd."  ,  "value"  :"WIPRO" } ,
                                      { "label"  :"Zee Tele"  ,  "value"  :"ZEETELE" } ,
                                      { "label"  :"Zee Entertainment Enterprises Ltd."  ,  "value"  :"ZEEL" } ,
                                      ], 
                             multi=True,value=['BPCL'],
                             style={"display": "block", "margin-left": "auto",'backgroundColor': '#d1e8e2', 
                                    "margin-right": "auto", "width": "56%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])





@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = stocks
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                     y=df[df["Symbol"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                     y=df[df["Symbol"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600, plot_bgcolor="#edf5e1",
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 12, 'label': 'Y', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (INR)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = stocks
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                     y=df[df["Symbol"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,  plot_bgcolor="#edf5e1",
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                       {'count': 12, 'label': 'Y', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=False)
