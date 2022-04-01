import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr

#Get Data from Yahoo Finance
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

#Measure port performance, returns the mean return and standard deviation
def portolioperformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns*weights)*Time
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
    return returns, std

stockList = ['TSLA', 'GOOG', 'MSFT', 'AMZN', 'FB', 'AAPL']
stocks = [stock  for stock in stockList] #Add .AX for Australian Stocks
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)
returns, meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
returns = returns.dropna()
weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)

#Add a column of port returns per day, according to randomized weights(port allocation)
returns['portfolio'] = returns.dot(weights)

#Historical VaR Function
def historicalVar(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVar, alpha=5)

    else:
        raise TypeError("Expected returns to be dataframe or series")

#Conditional VaR Function
def historicalCVar(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVar(returns, alpha=alpha) 
        return returns[belowVaR].mean()
    
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVar, alpha=5)

    else:
        raise TypeError("Expected returns to be dataframe or series")

#Predict port performance in the specified number of days (time)
Time = 1

print(returns)
print("Weights: ", weights)

VaR = -historicalVar(returns['portfolio'], alpha=5)*np.sqrt(Time)
CVaR = -historicalCVar(returns['portfolio'], alpha=5)*np.sqrt(Time)
pRet, pStd = portolioperformance(weights, meanReturns, covMatrix, Time)
InitialInvestment=1000

print("Initial Investment: ", InitialInvestment)
print("Expected Portfolio Return: ", round(InitialInvestment*pRet, 2))
print('Value at Risk 95th CI: ', round(InitialInvestment*VaR,2))
print('Conditional Value at Risk 95th CI: ', round(InitialInvestment*CVaR,2))
