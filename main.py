from flask import Flask,render_template,request,redirect,url_for
import yahoo_fin.stock_info as si
from yahoo_fin import options
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pickle
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from bs4 import BeautifulSoup
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing
import nltk
from keras.preprocessing.sequence import TimeseriesGenerator
import requests
nltk.download('vader_lexicon')
app=Flask(__name__)

credentials=pd.read_csv(r'C:\Users\kulka\PycharmProjects\Forecasting_sentiment\Data\creds.csv')

'''@app.route("/",methods=['POST','GET'])
def login():
    return render_template('login.html')'''

@app.route("/pred", methods=['GET', 'POST'])
def pred():
    comps = si.tickers_nifty50()
    return render_template('Forecast.html',comps=comps)

@app.route("/sent", methods=['GET', 'POST'])
def sent():
    comps = si.tickers_nifty50()
    return render_template('Sent.html',comps=comps)


@app.route("/", methods=['GET', 'POST'])
def login():
    user = request.form.get('UserName')
    user=str(user)
    password = request.form.get('password')
    password=str(password)
    output=''
    if len(credentials[(credentials['User'] == user) & (credentials['Password'] == password)])==0:
        return render_template('login.html',output='Please enter valid credentials!!')
    else:
        return redirect(url_for('pred'))
        #return render_template('login.html')

    #return render_template('login.html')



def create_dataset(data, look_back=1):
    data_X, data_y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), :]
        data_X.append(a)
        data_y.append(data[i + look_back, :])
    return np.array(data_X), np.array(data_y)

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    comps = si.tickers_nifty50()
    col='low'
    datecol='index'
    data = request.form['Stocks']
    data = str(data)
    # Loading the dataset

    dataset = si.get_data(data).reset_index()
    dataset = dataset.groupby([datecol]).sum([col]).reset_index()
    # Preprocessing the Data
    dataset = dataset.dropna()

    dataset = dataset[[datecol, col]]
    dataset[datecol] = pd.to_datetime(dataset[datecol])
    df = dataset.set_index(datecol)

    train = df.iloc[:round(len(df) * 0.8)]
    test = df.iloc[round(len(df) * 0.8):]
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    # define generator
    n_input = 3
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    X, y = generator[0]
    # We do the same thing, but now instead for 12 months
    n_input = 12
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(generator, epochs=3)

    # model = pickle.load(open('model.pkl', 'rb'))
    # predictions=pickled_model.predict(test_X)

    # loss_per_epoch = model.history.history['loss']
    last_train_batch = scaled_train[-12:]
    last_train_batch = last_train_batch.reshape((1, n_input, n_features))
    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(test) + 1):
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]

        # append the prediction into the array
        test_predictions.append(current_pred)

        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    true_predictions = scaler.inverse_transform(test_predictions)
    pred = scaler.inverse_transform(test_predictions[-1:])
    df1 = df.reset_index()
    nextdate = []
    for i in range(0, len(pred)):
        nextdate.append(max(df1[datecol]) + pd.DateOffset(i + 1))
    tempdf = pd.DataFrame()
    tempdf[datecol] = nextdate
    tempdf[col] = pred
    tempdf = pd.concat([df1[[datecol, col]], tempdf], ignore_index=True)
    tempdf[datecol] = tempdf[datecol]
    out=list(tempdf[col].tail(1))[0]
    Predicted = 'Predicted Price tommorrow:'

    return render_template('Forecast.html',Predicted=Predicted,prediction_text= out,comps=comps)


def Subjectivity(text):
  return TextBlob(text).sentiment.subjectivity

def Polarity(text):
  return  TextBlob(text).sentiment.polarity

@app.route("/impact", methods=['GET', 'POST'])
def impact():
    comps = si.tickers_nifty50()
    col='low'
    datecol='index'
    data = request.form['Stocks']
    data = str(data)
    df_num = si.get_data(data).reset_index()
    url = 'https://www.moneycontrol.com/news/business/stocks/'
    content = requests.get(url)
    HTMLCON = content.content
    soup = BeautifulSoup(HTMLCON, 'html.parser')
    news = soup.findAll('ul', {'id': 'cagetory'})
    headline = []
    for i in range(0, len(news)):
        headline.append(news[i].get_text())
    df_text = pd.read_csv(r'C:\Users\kulka\PycharmProjects\Forecasting_sentiment\Data\raw_partner_headlines.csv').tail(len(df_num))

    df_text = df_text[['date', 'headline']].sort_values(by='date', ascending=True)

    df_text.rename(columns={'date': 'publish_date', 'headline': 'headline_text'}, inplace=True)
    df_text.reset_index(inplace=True)
    df_text.drop(columns='index', inplace=True)

    df_text.loc[len(df_text.index)] = [pd.to_datetime(date.today(), format='%Y-%m-%d %H:%M:%S'), headline[0]]
    df_text["publish_date"] = pd.to_datetime(df_text["publish_date"], format='%Y-%m-%d %H:%M:%S')
    df_text['headline_text'] = df_text.groupby(['publish_date']).transform(lambda x: ' '.join(x))
    df_text.reset_index(inplace=True, drop=True)
    df_text.replace("[^a-zA-Z']", " ", regex=True, inplace=True)
    df_text['Subjectivity'] = df_text['headline_text'].apply(Subjectivity)
    df_text['Polarity'] = df_text['headline_text'].apply(Polarity)

    snt = SentimentIntensityAnalyzer()
    df_text['Compound'] = [snt.polarity_scores(v)['compound'] for v in df_text['headline_text']]
    df_text['Negative'] = [snt.polarity_scores(v)['neg'] for v in df_text['headline_text']]
    df_text['Neutral'] = [snt.polarity_scores(v)['neu'] for v in df_text['headline_text']]
    df_text['Positive'] = [snt.polarity_scores(v)['pos'] for v in df_text['headline_text']]
    df_text1 = df_text.tail(1)
    df_text = df_text.head(len(df_text) - 1)

    df_num.rename(columns={'index': 'Date'}, inplace=True)
    df_num["Date"] = pd.to_datetime(df_num["Date"], format='%Y-%m-%d')
    df_num.ffill(inplace=True)
    close = df_num['close']

    ma = close.rolling(window=50).mean()
    std = close.rolling(window=50).std()

    # split the data to train and test
    train = df_num[:round(len(df_num) * 0.8)]
    test = df_num[round(len(df_num) * 0.8):]


    merge = df_text
    data = merge[['Subjectivity', 'Polarity', 'Compound', 'Negative', 'Neutral', 'Positive']]
    X = data[:len(df_num)]
    y = df_num['close']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    out=rf.predict(df_text1[['Subjectivity',	'Polarity',	'Compound',	'Negative',	'Neutral',	'Positive']])
    Predicted = 'Predicted Price tommorrow:'
    return render_template('Sent.html', Predicted=Predicted, prediction_text=out[0], comps=comps)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30006, debug=True)

