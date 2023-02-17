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
	ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
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

col1, col2 = st.columns(2)
with col1:
    start_date = st.text_input("Start Date, e.g. 2018-01-01")
with col2:
    end_date = st.text_input("End Date, e.g. 2023-02-01") # it defaults to current date
 
tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas \
 WITHOUT spaces, e.g. "TCB","SSI","VHC","VHM","HBC","FPT","HPG","HVN","TRA","POW"', '').upper()
tickers = tickers_string.split(',')




try:
	loader = DataLoader(tickers, start_date ,end_date, minimal=True, data_source = "cafe")   
	data= loader.download()
	data=data.stack()
	data=data.reset_index()     
	stocks_df = data.pivot_table(values = 'adjust', index = 'date', columns = 'Symbols').dropna()
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
	fig = plot_efficient_frontier_and_max_sharpe(mu, S)
	fig_efficient_frontier = BytesIO()
	fig.savefig(fig_efficient_frontier, format="png")
	
	# Get optimized weights
	ef = EfficientFrontier(mu, S)
	ef.max_sharpe(risk_free_rate=0.02)
	weights = ef.clean_weights()
	expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
	weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
	weights_df.columns = ['weights']
	
	# Calculate returns of portfolio with optimized weights
	stocks_df['Optimized Portfolio'] = 0
	for ticker, weight in weights.items():
		stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
	
	# Plot Cumulative Returns of Optimized Portfolio
	fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
	
	# Display everything on Streamlit
	st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))	
	st.plotly_chart(fig_cum_returns_optimized)
	
	st.subheader("Optimized Max Sharpe Portfolio Weights")
	st.dataframe(weights_df)
	
	st.subheader("Optimized Max Sharpe Portfolio Performance")
	st.image(fig_efficient_frontier)
	
	st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
	st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
	st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
	
	st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
	st.plotly_chart(fig_price)
	st.plotly_chart(fig_cum_returns)
except Exception as e:
	st.write(e)
	st.write('Enter correct stock tickers to be included in portfolio separated\
              commas WITHOUT spaces, e.g. TCB,HPG,SSI,MSN  and hit Enter.')	
