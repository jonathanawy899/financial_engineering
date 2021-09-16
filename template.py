import numpy as np
import pandas as pd
import datetime as dt

datafilename = "./stock_data_f_5min.xlsx"
wgtfilename = "./stock_weights_f.xlsx"

INIT_FUND = 1e8
print(f"Initial fund available: {INIT_FUND}")

stock_data = pd.read_excel(datafilename, header=[0,1], index_col=0, parse_dates=True)
wgt_data = pd.read_excel(wgtfilename, index_col=0)

# Stock list:
stock_list = list(wgt_data.index)
print(stock_list)
# Normalize weights
stock_weights = (wgt_data['weight']/(wgt_data['weight'].sum())).to_dict()
print(stock_weights)

# Max position
max_pos_ratio = 0.5 # Use 50% of available fund and allocate to each of the 20 stocks by their weight defined above
# PX_column
PX_COL = "close" # use close px of every 5min (we use the 5min candlestick) to enter our position

# strategy parameters
curr_fund = INIT_FUND
start = True

# the dataframe to store allocation
position = pd.DataFrame()
unused_fund = pd.Series(dtype=float)

for ts, datarow in stock_data.iterrows():
    # calculate available fund:
    if start:
        # skip since we already defined the curr_fund in the preparation stop
        start = False
    else:
        # get previous allocation
        prev_pos = position.loc[prev_ts] 
        curr_fund = (prev_pos*datarow.unstack()[PX_COL]).sum() + fund_left  # "fund_left" is the fund that is not used in the previous timestep
    #####################################################################################
    ###Trading Stratedy###
    for stock in stock_list:
        fund_of_stock = max_pos_ratio*curr_fund*stock_weights[stock]
        num_shares = np.floor(fund_of_stock/datarow[stock][PX_COL])
        position.at[ts, stock] = num_shares
    
    ######################################################################################
    fund_left = curr_fund - (position.loc[ts] * datarow.unstack()[PX_COL]).sum() # fund not used.
    unused_fund.at[ts] = fund_left

    prev_ts = ts


used_fund = (position*stock_data.loc[:, (slice(None), PX_COL)].droplevel('field',axis=1)).sum(axis=1)
total_fund = used_fund + unused_fund
# evaluate using quantstats

import quantstats as qs

# Sharpe ratio:
sharpe_ratio = qs.stats.sharpe(total_fund/INIT_FUND)
print(f"Sharpe ratio: {sharpe_ratio}")

qs.plots.snapshot(total_fund, title='weighted distribution performance',figsize=(12,8), lw=2, fontname="DejaVu Sans")