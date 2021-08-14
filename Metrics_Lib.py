import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import date

def standarization(data):
    if 'TICKER' in data:
        data = data.drop(['TICKER'], axis=1)
    if 'CLUSTERS' in data:
        data = data.drop(['CLUSTERS'], axis=1)
    data = ((data - (data.mean())) / (data.std()))**2
    return data

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

def cleaner(string):
        characters_to_remove = "['']"
        string = str(string)
        for character in characters_to_remove:
              string = string.replace(character, "")
        return string

def replace_item(lst, to_replace, replace_with):
    return sum((replace_with if i==to_replace else [i] for i in lst), [])

def beta_calculator(returns):
        beta = (returns.cov()[composite])/(returns[composite].var())
        return beta

def null_portfolio(prices, portfolio, selected_portfolio):
    while prices.loc[:, prices.isna().any()].shape[1] >= 1:        
        n = prices.loc[:, prices.isna().any()].shape[1] #number of nan columns        
        old = cleaner((prices.loc[:, prices.isna().any()].keys()).tolist()) #sacar el nombre de las nan columns     
        prices = prices.dropna(axis = 1) #remove nan from prices        
        selected_portfolio = selected_portfolio[~selected_portfolio.TICKER.isin(portfolio)] #clean the old selected_data with portfolio tickers       
        replaces = (np.random.choice(selected_portfolio['TICKER'], n, replace=False)).tolist() #new portfolio tickers        
        prices[replaces] = yf.download(replaces, start = start, end = today, interval="1d" )['Adj Close']   
        portfolio = replace_item(portfolio, old, replaces)
        Last_price = yf.download(portfolio, today)['Adj Close']
    return Last_price, prices, portfolio
    
def Risk(portfolio, composite, prices, Last_price, Risk_Free_Rate, Shares, Years):
    Ponderation = ((Last_price * Shares).apply(lambda x: x/((Last_price * Shares).sum(axis=1)), 0)).to_numpy()
    Average_Daily_Return = (prices.pct_change())[portfolio].mean()
    Portfolio_Daily_Return = np.dot(np.squeeze(Average_Daily_Return),np.squeeze(Ponderation.T))
    Portfolio_Annual_Return = ((1 + Portfolio_Daily_Return)**252)-1
    Portfolio_Daily_Risk = (np.sqrt(np.matmul(Ponderation,np.matmul((((prices.pct_change())[portfolio].cov()[portfolio])),Ponderation.T)))).to_numpy()
    Portfolio_Annual_Risk = Portfolio_Daily_Risk * np.sqrt(252)
    Investment = (Last_price * Shares).sum(axis = 1)
    return Ponderation, Average_Daily_Return, Portfolio_Daily_Risk, Portfolio_Annual_Return, Portfolio_Annual_Risk, Investment

def VaR(prices, portfolio, Last_price, Shares, Confidence_Interval):
    Risk_Factor = np.exp(prices[portfolio].pct_change().dropna()).multiply(Last_price.squeeze(),axis=1)
    Market_Value_Scenario = Shares * Risk_Factor
    Market_Value_Scenario['TOTAL'] = Market_Value_Scenario.sum(axis = 1)
    Market_Value_Today = Last_price * Shares
    Market_Value_Today['TOTAL'] = Market_Value_Today.sum(axis = 1)
    Profit_Loses = Market_Value_Today.squeeze() - Market_Value_Scenario
    Daily_VaR = pd.DataFrame([np.quantile(Profit_Loses[portfolio[i]] ,1-Confidence_Interval) for i in range(0, len(portfolio))])
    Daily_VaR.set_index([portfolio], inplace = True)
    Daily_VaR.loc['TOTAL'] = np.quantile(Profit_Loses['TOTAL'] ,1-Confidence_Interval)
    Percent_VaR = np.abs(Daily_VaR.squeeze()/Market_Value_Today.T.squeeze())
    Diversification = Daily_VaR.loc[:,0].sum()
    return Daily_VaR, Percent_VaR, Diversification

def stocks_summary(portfolio, composite, Years):
    today = date.today()
    start = today.replace(year=today.year - Years)
    prices = yf.download(portfolio, start = start, end = today, interval="1d" )['Adj Close']
    prices[composite] = yf.download(composite, start = start, end = today, interval="1d" )['Adj Close']
    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    prices = prices.fillna(prices.mean())
    if prices.loc[:, prices.isna().any()].shape[1] >= 1:
        Last_price, prices, portfolio = null_portfolio(prices, portfolio, selected_portfolio)
    else:
    Last_price = yf.download(portfolio, start = start, end = today)['Adj Close'] 
    return prices, Last_price

def CAPM(portfolio, composite, Risk_Free_Rate, Shares, Years):
    today = date.today()
    start = today.replace(year=today.year - Years)
    prices = yf.download(portfolio, start = start, end = today, interval="1d" )['Adj Close']
    prices[composite] = yf.download(composite, start = start, end = today, interval="1d" )['Adj Close']
    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    prices = prices.fillna(prices.mean())
    if prices.loc[:, prices.isna().any()].shape[1] >= 1:
        Last_price, prices, portfolio = null_portfolio(prices, portfolio, selected_portfolio)
    else:
    Last_price = yf.download(portfolio, start = start, end = today)['Adj Close']    
    Ponderation = ((Last_price * Shares).apply(lambda x: x/((Last_price * Shares).sum(axis=1)), 0)).to_numpy()
    Average_Daily_Return = (prices.pct_change())[portfolio].mean()
    Portfolio_Daily_Return = np.dot(np.squeeze(Average_Daily_Return),np.squeeze(Ponderation.T))
    Portfolio_Annual_Return = ((1 + Portfolio_Daily_Return)**252)-1
    Portfolio_Daily_Risk = (np.sqrt(np.matmul(Ponderation,np.matmul((((prices.pct_change())[portfolio].cov()[portfolio])),Ponderation.T)))).to_numpy()
    Portfolio_Annual_Risk = Portfolio_Daily_Risk * np.sqrt(252)
    Investment = (Last_price * Shares).sum(axis = 1)   
    Daily_Composite_Return = (prices.pct_change())[composite].mean()
    Stock_Beta = pd.DataFrame(beta_calculator(prices.pct_change())).drop(composite,axis=0)
    Portfolio_Beta = np.dot(Stock_Beta.T,Ponderation.T)
    Market_Return = ((1 + Daily_Composite_Return)**252)-1
    CAPM = Risk_Free_Rate + Stock_Beta * (Market_Return - Risk_Free_Rate)
    Portfolio_CAPM = Risk_Free_Rate + Portfolio_Beta * (Market_Return - Risk_Free_Rate)
    return CAPM, Portfolio_CAPM
