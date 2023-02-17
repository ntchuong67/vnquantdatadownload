# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 18:08:04 2023

@author: admin
"""

import pandas as pd
import numpy as np

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import hierarchical_portfolio
from pypfopt import black_litterman
from pypfopt import plotting
import streamlit as st
from datetime import datetime
from vnquant import *
import matplotlib.pyplot as plt
import copy
import plotly.express as px


def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*100
	fig = px.line(daily_cum_returns, title=title)
	return fig
	
def plot_efficient_frontier_and_max_sharpe(mu, S): 
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S)
	fig, ax = plt.subplots(figsize=(6,4))
	ef_max_sharpe = copy.copy(ef)
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe()
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig

st.set_page_config(page_title = "Stock Portfolio Optimizer - developed by Nguyen Tien Chuong", layout = "wide")
st.header("Stock Portfolio Optimizer")

col0, col1, col2 = st.columns(3)
with col0:
    start_date = st.text_input("Start Date, e.g. 2018-01-01")
with col1:
    trial_date = st.text_input("End Traning Date, e.g. 2022-01-01")
with col2:
    end_date = st.text_input("End Date, e.g. 2023-02-01") # it defaults to current date
 
tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas \
 WITHOUT spaces, e.g. TCB,HPG,SSI,MSN', '').upper()
tickers = tickers_string.split(',')




try:
	#trial data
	loader = DataLoader(tickers, start_date ,trial_date, minimal=True, data_source = "vnd")   
	data= loader.download()
	data=data.stack()
	data=data.reset_index()     
	stocks_df = data.pivot_table(values = 'close', index = 'date', columns = 'Symbols').dropna()
	#full data
	loader2 = DataLoader(tickers, start_date ,end_date, minimal=True, data_source = "vnd")   
	data2= loader.download()
	data2=data2.stack()
	data2=data2.reset_index()     
	full_stocks_df = data2.pivot_table(values = 'close', index = 'date', columns = 'Symbols').dropna()
	
	st.dataframe(stocks_df)
	# Plot Individual Stock Prices
	fig_price = px.line(stocks_df, title='Price of Individual Stocks')
	# Plot Individual Cumulative Returns
	fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
	# Calculatge and Plot Correlation Matrix between Stocks
	corr_df = stocks_df.corr().round(2)
	fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')
		
	# Calculate expected returns and sample covariance matrix for portfolio optimization later
	mu = expected_returns.mean_historical_return(stocks_df)
	S = risk_models.sample_cov(stocks_df)
	
	# Plot efficient frontier curve
	#fig = plot_efficient_frontier_and_max_sharpe(mu, S)
	#fig_efficient_frontier = BytesIO()
	#fig.savefig(fig_efficient_frontier, format="png")
	
	# Get optimized weights
	ef = EfficientFrontier(mu, S)
	ef.max_sharpe()
	weights = ef.clean_weights()
	expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
	weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
	weights_df.columns = ['weights']
	
	
	
	#=======HRP=================
	stocks_df2 = data.pivot_table(values = 'close', index = 'date', columns = 'Symbols').dropna()
	full_stocks_df2 =  data2.pivot_table(values = 'close', index = 'date', columns = 'Symbols').dropna()
	
	returns = expected_returns.returns_from_prices(stocks_df2, log_returns=False)
	hierarchical_portfolio.HRPOpt(returns,S)
	hrp = hierarchical_portfolio.HRPOpt(returns,risk_models.sample_cov(stocks_df2))
	weight_hrp = hrp.optimize()
	expected_annual_return_hrp, annual_volatility_hrp, sharpe_ratio_hrp = hrp.portfolio_performance()
	
	#====PLOTTING========================================
	# Display everything on Streamlit
	st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
	st.plotly_chart(fig_price)
	st.plotly_chart(fig_cum_returns)
	#Weights of portfolios
	col3, col4 = st.columns(2)
	with col3:
		st.subheader("Optimized Max Sharpe Portfolio Weights")
		st.dataframe(weights_df)
	with col4:
		st.subheader("Optimized HRP Portfolio Weights")
		st.dataframe(weight_hrp)
	#st.subheader("Optimized Max Sharpe Portfolio Performance")
	#st.image(fig_efficient_frontier)
	col5, col6 = st.columns(2)
	with col5:
		st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
		st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
		st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
	with col6:
		st.subheader('Expected annual return: {}%'.format((expected_annual_return_hrp*100).round(2)))
		st.subheader('Annual volatility: {}%'.format((annual_volatility_hrp*100).round(2)))
		st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio_hrp.round(2)))
	
	# Calculate returns of portfolio with optimized weights
	full_stocks_df['Optimized Portfolio Max Sharpe'] = 0
	full_stocks_df2['Optimized Portfolio HRP'] = 0
	for ticker, weight in weights.items():
		full_stocks_df['Optimized Portfolio Max Sharpe'] += full_stocks_df[ticker]*weight
	for ticker, weight in weight_hrp.items():
		full_stocks_df2['Optimized Portfolio HRP'] += full_stocks_df2[ticker]*weight
	full_stocks_df['Optimized Portfolio HRP']= full_stocks_df2['Optimized Portfolio HRP']	
	# Plot Cumulative Returns of Optimized Portfolio
	fig_cum_returns_optimized = plot_cum_returns(full_stocks_df[['Optimized Portfolio Max Sharpe','Optimized Portfolio HRP']], 'Cumulative Returns of Optimized Portfolio Starting with $100')
	
	st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))	
	st.plotly_chart(fig_cum_returns_optimized)
except Exception as e:
	st.write(e)
	st.write('Enter correct stock tickers to be included in portfolio separated\
              commas WITHOUT spaces, e.g. TCB,HPG,SSI,MSN  and hit Enter.')	
