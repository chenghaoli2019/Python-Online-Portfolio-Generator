from IPython.display import display, Math, Latex

import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import bs4 as bs
import requests
import yfinance as yf
import datetime

def main():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    tickers = [s.replace('\n', '') for s in tickers]
    df = pd.DataFrame (tickers, columns = ['0'])

    ticker_list=df

# Assign Start and End date for the analysis
    start_date="2022-11-02"
    end_date= "2022-12-30"

# Determine the minimum number of trading days required to be considered
    apple =yf.Ticker("AAPL")
    apple_hist = apple.history(start= start_date, end= end_date)
    apple_hist.dropna(inplace=True)
    min_data_len = len(apple_hist.index)
# Extract first trading day
    apple_hist.reset_index(inplace=True)
    first_trading_day = str(apple_hist["Date"][0])
    first_trading_day = first_trading_day[0:10]
    ticker_list=ticker_list.drop_duplicates()
    ticker_list=ticker_list.replace('N/A',np.NaN)
    ticker_list=ticker_list.dropna()
    ticker_list.reset_index(drop=True, inplace=True)
# Create empty column called score
    ticker_list['score']=''
    threads = []
    for i in range(0,len(ticker_list)):
    # Create a new thread
        t = threading.Thread(target=eval_ticker, args=[ticker_list,i,start_date, end_date])
    
    # Start thread
        t.start()
    
    # Add thread to thread list
        threads.append(t)
    for thread in threads:
        thread.join()
    ticker_list.sort_values(by=['score'], ascending=False, inplace=True)
    ticker_list.reset_index(drop=True,inplace=True)
    ticker_list.dropna(inplace=True)
    # Keep 25 most risky stocks
    ticker_list = ticker_list.head(25)

# Convert tickers into list
    risky_tickers = list(ticker_list['0'])

# Generate dictionary of stock prices
    stock_price_dict = generate_stock_df_dict(risky_tickers,start_date)

# Generate 25 risky portfolios
    portfolios = generate_portfolios(risky_tickers, stock_price_dict)

    portfolio_stds = []
    #Applys buildportfolio to every list of tickers in portfolios
    for n in portfolios:
        buildportfolio(n,stock_price_dict,portfolio_stds)
    
        finalport_tickerslst = portfolios[portfolio_stds.index(max(portfolio_stds))]

        finalport_pricelst = []

# Defines finalport_pricelst as the price of each ticker at the required date
    for i in range(len(finalport_tickerslst)):
        finalport_pricelst.append(stock_price_dict[finalport_tickerslst[i]].loc["2022-11-25", "Close"])
    
    finalport_shareslst = []

#Defines finalport_shareslst as the number of shares of each stock, based on finalport_pricelst and the position of the ticker in the finalport_tickerslst
    for n in range(len(finalport_tickerslst)):
        value = -1
        if n <= 1:
            value = 125000
        elif n == 2:
            value = 62500
        else:
            value = 62500/3
        finalport_shareslst.append(value/finalport_pricelst[n])
        #Creates a dictionary with the data above
    finalport = {"Ticker" : finalport_tickerslst,
             "Price" : finalport_pricelst,
             "Shares" : finalport_shareslst}

#Creates the index for the portfolio
    index_list = list(range(1, 13))

#Creates the dataframe from the data in finalport
    Portfolio_Final = pd.DataFrame(finalport, index=index_list)

#Adds a column for Value using price * shares
    Portfolio_Final["Value"] = Portfolio_Final.Price * Portfolio_Final.Shares

#Adds a column for Weight based oln the value of each stock
    Portfolio_Final["Weight"] = Portfolio_Final.Value / 500000 * 100

    Stocks_Final = Portfolio_Final[["Ticker", "Shares"]]

#Exports the dataframe to a csv file
    print(Stocks_Final)
    return Stocks_Final
# Calculates and insert score for each ticker in the dataframe
# Consumes Ticker list dataframe, index, start and end date
def eval_ticker(ticker_list,i,start_date, end_date):
    ticker_name=ticker_list.iloc[i,0]
    ticker=yf.Ticker(ticker_name)
    company=ticker.history(start= start_date, end= end_date)
    company=company["Close"]
    company=company.pct_change()
    company=company.dropna()
    standard_deviation=company.std()
    score=standard_deviation
    ticker_list['score'][i]=score



# Start Threading in a loop

# Stop all threads in list




# Reset the index

def generate_stock_df_dict(ticker_lst,start_date):
    # Variables
    hist_interval = "1d"  
    df_dict = {}
    
    # Loop through each ticker in ticker_lst
    for ticker in ticker_lst:
        stock = yf.Ticker(ticker)
        
        # Get stock prices
        stock_hist = stock.history(start=start_date, interval=hist_interval)
        
        # Drop empty rows
        stock_hist.dropna(inplace=True)
        
        # Add DF with just Close column to dict
        df_dict[ticker] = stock_hist[["Close"]]
        
    # Return dict
    return df_dict

def find_corr(ticker, second_ticker, df_dict):
    # Variables
    ls = "_" + ticker
    rs = "_" + second_ticker
    
    # Join the price DataFrames for the two stocks
    prices = df_dict[ticker].join(df_dict[second_ticker], lsuffix=ls, rsuffix=rs)
    
    # Drop empty rows
    prices.dropna(inplace=True)
    
    # Get daily returns and drop first row
    daily_returns = prices.pct_change()
    daily_returns = daily_returns.iloc[1:]
    
    # Extract correlation
    returns_corr = daily_returns.corr().iloc[0, 1]
    
    # Return correlation
    return returns_corr

def generate_portfolios(ticker_lst, df_dict):
    # Create empty list
    portfolios = []
    
    # Loop through each ticker in ticker_lst
    for ticker in ticker_lst:
        # Copy ticker_lst to avoid reference problems
        ticker_lst_copy = ticker_lst.copy()
        
        # Remove current ticker from copy
        ticker_lst_copy.remove(ticker)
        
        # Create a new DF where the index is ticker_lst_copy
        corr_df = pd.DataFrame(index=ticker_lst_copy)
        
        # Name the index Stock
        corr_df.index.name = "Stock"
        
        # Create a new column called Corr where each value is initially -2
        corr_df["Corr"] = -2
        
        # Loop through each ticker in ticker_lst_copy
        for second_ticker in ticker_lst_copy:
            # Find correlation between tickers in outer and inner for loops
            corr = find_corr(ticker, second_ticker, df_dict)
            
            # Place correlation value in appropriate location in DF
            corr_df.loc[second_ticker, "Corr"] = corr
        
        # Reset index
        corr_df.reset_index(inplace=True)
        
        # Sort DF values by Corr column from most correlated to least correlated
        corr_df.sort_values(by="Corr", ascending=False, inplace=True)
        
        # Create a new one element list containing ticker in outer for loop
        current_portfolio = [ticker]
        
        # Extract 11 most correlated stocks as a list
        additional_stocks = list(corr_df.head(11).Stock)
        
        # Extend the one element list to have the correlated stocks
        current_portfolio.extend(additional_stocks)
        
        # Append current portfolio to list of portfolios
        portfolios.append(current_portfolio)
    
    # Return list of portfolios
    return portfolios
def buildportfolio(lst,stock_price_dict,portfolio_stds):
    portfolio = pd.DataFrame()
    
    #Determines how much of the portfolio the stock will be worth, based on its position in the list
    for x in range(len(lst)):
        if x <= 1:
            investment = 125000
        elif x == 2:
            investment = 62500
        else:
            investment = 62500/3
        
        #Gets closing price value from the dictionary defined earlier
        portfolio[lst[x]] = stock_price_dict[lst[x]]
        
        #Defines the number of shares based on the stock price from the first day
   
        try:
            num_shares = investment / portfolio.loc[first_trading_day, lst[x]]
        except:
            num_shares = investment / portfolio[lst[x]].iloc[0]
        
        #Multiplies closing value by number of shares 
        portfolio[lst[x]] = portfolio[lst[x]] * num_shares
        
    #Drops NaN values
    portfolio = portfolio.dropna()
    
    #Adds all stocks into a final column to track the portfolio value over time
    portfolio["portfolio value"] = portfolio.sum(axis = 1)
    
    #Adds the standard deviation of the portfolio to the portfolio_stds
    portfolio_stds.append(portfolio['portfolio value'].std())
    

    
main()