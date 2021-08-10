import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from sklearn.cluster import KMeans

def cleaner(string):
    characters_to_remove = "['']"
    string = str(string)
    for character in characters_to_remove:
          string = string.replace(character, "")
    return string

def replace_item(lst, to_replace, replace_with):
    return sum((replace_with if i==to_replace else [i] for i in lst), [])

def clustering(data):
    z = 4
    wcss = []
    for i in range(1, z):
        kmeans = KMeans(n_clusters = i, max_iter = 300)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    clustering = KMeans(n_clusters = z, max_iter = 300)
    clustering.fit(data)
    data['CLUSTERS'] = clustering.labels_
    return data

def standardization(data):
    if 'TICKER' in data:
        data = data.drop(['TICKER'], axis=1)
    if 'CLUSTERS' in data:
        data = data.drop(['CLUSTERS'], axis=1)
    U = data.mean()
    desvest = data.std()
    data = ((data - U) / desvest)**2
    return data

def beta_calculator(returns):
    stocks_cov = returns.cov()[composite]
    mkt_var = returns[composite].var()
    beta = stocks_cov/mkt_var
    return beta

def stocks_summary(portfolio, composite, Years):
    today = date.today()
    start = today.replace(year=today.year - Years)
    prices = yf.download(portfolio, start = start, end = today, interval="1d" )['Adj Close']
    prices[composite] = yf.download(composite, start = start, end = today, interval="1d" )['Adj Close']
    Last_price = yf.download(portfolio, today)['Adj Close']
    prices = prices.fillna(method='bfill')
    return prices, Last_price

def risk(portfolio, composite, prices, Last_price, Risk_Free_Rate, Shares, Years):
    net_returns = prices.pct_change()
    Market_Value = Last_price * Shares
    Total_Market_Value = Market_Value.sum(axis=1)
    Ponderation = (Market_Value.apply(lambda x: x/Total_Market_Value, 0)).to_numpy()
    Daily_Composite_Return = net_returns[composite].mean()
    Average_Daily_Return = net_returns[portfolio].mean()
    Portfolio_Daily_Return = np.dot(np.squeeze(Average_Daily_Return),np.squeeze(Ponderation.T))
    Portfolio_Annual_Return = ((1 + Portfolio_Daily_Return)**252)-1
    Cov_Matrix = (net_returns[portfolio].cov()[portfolio])
    Portfolio_Daily_Risk = (np.sqrt(np.matmul(Ponderation,np.matmul(Cov_Matrix,Ponderation.T)))).to_numpy()
    Portfolio_Annual_Risk = Portfolio_Daily_Risk * np.sqrt(252)
    return Ponderation, Average_Daily_Return, Portfolio_Daily_Risk, Portfolio_Annual_Return, Portfolio_Annual_Risk

def CAPM(portfolio, composite, Risk_Free_Rate, Shares, Years):
    prices = yf.download(portfolio, start = start, end = today, interval="1d" )['Adj Close']
    prices[composite] = yf.download(composite, start = start, end = today, interval="1d" )['Adj Close']
    Last_price = yf.download(portfolio, today)['Adj Close']
    prices = prices.fillna(method='bfill')
    net_returns = prices.pct_change()
    Market_Value = Last_price * Shares
    Total_Market_Value = Market_Value.sum(axis=1)
    Ponderation = (Market_Value.apply(lambda x: x/Total_Market_Value, 0)).to_numpy()
    Daily_Composite_Return = net_returns[composite].mean()
    Average_Daily_Return = net_returns[portfolio].mean()
    Portfolio_Daily_Return = np.dot(np.squeeze(Average_Daily_Return),np.squeeze(Ponderation.T))
    Portfolio_Annual_Return = ((1 + Portfolio_Daily_Return)**252)-1
    Cov_Matrix = (net_returns[portfolio].cov()[portfolio])
    Portfolio_Daily_Risk = (np.sqrt(np.matmul(Ponderation,np.matmul(Cov_Matrix,Ponderation.T)))).to_numpy()
    Portfolio_Annual_Risk = Portfolio_Daily_Risk * np.sqrt(252)
    Daily_Composite_Return = net_returns[composite].mean()
    Stock_Beta = pd.DataFrame(beta_calculator(net_returns)).drop(composite,axis=0)
    Portfolio_Beta = np.dot(Stock_Beta.T,Ponderation.T)
    Market_Return = ((1 + Daily_Composite_Return)**252)-1
    CAPM = Risk_Free_Rate + Stock_Beta * (Market_Return - Risk_Free_Rate)
    Portfolio_CAPM = Risk_Free_Rate + Portfolio_Beta * (Market_Return - Risk_Free_Rate)
    return CAPM, Portfolio_CAPM

def null_portfolio(prices, portfolio, selected_portfolio):
    while prices.loc[:, prices.isna().any()].shape[1] >= 1:
        n = prices.loc[:, prices.isna().any()].shape[1] #number of nan columns
        old = (prices.loc[:, prices.isna().any()].keys()).tolist() #sacar el nombre de las nan columns
        old = cleaner(old)
        prices = prices.dropna(axis = 1) #remove nan from prices
        name = prices.loc[:, prices.isna().any()].head()
        selected_portfolio = selected_portfolio[~selected_portfolio.TICKER.isin(portfolio)] #clean the old selected_data with portfolio tickers
        replaces = (np.random.choice(selected_portfolio['TICKER'], n, replace=False)).tolist() #new portfolio tickers
        prices[replaces] = yf.download(replaces, start = start, end = today, interval="1d" )['Adj Close']
        portfolio = replace_item(portfolio, old, replaces)
        Last_price = yf.download(portfolio, today)['Adj Close']
    return Last_price, prices, portfolio
