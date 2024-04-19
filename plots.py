import utils as ut
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import combinations
import seaborn as sns
import pandas as pd
import numpy as np


def figure_1(df_price_data,df_comms_ret):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)

    # First two rows for TTF, POW, EUA, and their returns
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    # Plotting the TTF, POW, and EUA prices and their returns
    titles = ['TTF', 'POW', 'EUA', 'Return TTF', 'Return POW', 'Return EUA']
    data_keys = ['ttf_close', 'power_close', 'eua_close', 'ttf_ret', 'pow_ret', 'eua_ret']
    for i, (title, key) in enumerate(zip(titles, data_keys)):
        axs[i].plot(df_price_data.index[1:] if i < 3 else df_comms_ret.index, 
                    df_price_data[key][1:] if i < 3 else df_comms_ret[key])
        axs[i].set_title(title)
        axs[i].xaxis.set_major_locator(mdates.YearLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Adjust layout for the plots above
    plt.tight_layout()

    # Third row for EWMA correlations, spanning all columns
    ax_ewma = fig.add_subplot(gs[2, :])

    # Calculating and plotting EWMA correlations
    combinations_cols = list(combinations(['ttf_ret', 'pow_ret', 'eua_ret'], 2))

    for col in combinations_cols:
        r_temp = df_comms_ret[list(col)]
        r_centered = r_temp - np.mean(r_temp, axis=0)
        rhoewma = ut.ewma(r_centered.values, 0.95)
        sns.lineplot(x=df_comms_ret.index, y=rhoewma.T[0], ax=ax_ewma, label=f'{col[0][:-4].upper()} & {col[1].upper()} Correlation')

    # Set the title and labels for the EWMA plot
    ax_ewma.set_title('EWMA Correlations')
    ax_ewma.set_xlabel('Date')
    ax_ewma.set_ylabel('Correlation')

    # Adjust the overall layout
    plt.tight_layout()
    return fig

def figure_2(df_comms_ret,chi_1,chi_2):

    # Create a figure with a 2x3 grid
    fig, axs = plt.subplots(1, 2, figsize=(18, 12))

    # Flatten the axs array to simplify indexing
    axs = axs.flatten()

    # First subplot: TTF
    axs[0].plot(df_comms_ret.index[2:],chi_1[1][1:])
    axs[0].set_title('Gaussian two regimes')
    axs[0].xaxis.set_major_locator(mdates.YearLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Second subplot: EUA
    axs[1].plot(df_comms_ret.index[2:],chi_2[1][1:])
    axs[1].set_title('Student t two regimes')
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))



    # Adjust spacing between subplots
    plt.tight_layout()

    return fig

def figure_3(df_comms_ret,condcorr_stu_ar,corrstudcc):
    # Create a figure with a 2x3 grid
    fig, axs = plt.subplots(1, 3, figsize=(18, 12))

    # Flatten the axs array to simplify indexing
    axs = axs.flatten()

    # First subplot: TTF

    axs[0].plot(df_comms_ret.index[2:],condcorr_stu_ar[:,0,1][1:-1],label = 'MS CCC-GARCH Stu')
    axs[0].plot(df_comms_ret.index[2:],corrstudcc[:,0,1][1:-2],c ='gray',label = 'DCC Stu')
    axs[0].set_title('Conditional correlations student t innovations TTF EUA')
    axs[0].xaxis.set_major_locator(mdates.YearLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0].legend()
    # Second subplot: EUA
    axs[1].plot(df_comms_ret.index[2:],condcorr_stu_ar[:,0,2][1:-1],label = 'MS CCC-GARCH Stu')
    axs[1].plot(df_comms_ret.index[2:],corrstudcc[:,0,2][1:-2],c ='gray',label = 'DCC Stu')
    axs[1].set_title('Conditional correlations student t innovations TTF OIL')
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[1].legend()
    axs[2].plot(df_comms_ret.index[2:],condcorr_stu_ar[:,1,2][1:-1],label = 'MS CCC-GARCH Stu')
    axs[2].plot(df_comms_ret.index[2:],corrstudcc[:,1,2][1:-2],c ='gray',label = 'DCC Stu')
    axs[2].set_title('Conditional correlations student t innovations EUA OIL')
    axs[2].xaxis.set_major_locator(mdates.YearLocator())
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[2].legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    return fig

def figure_4(var_list_stu_skewed_t_5,var_list_stu_skewed_t_95,var_list_stu,
             var_list_norm,var_list_stu_single,var_list_stu_dcc,rret_short,rret_long):
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(26, 12)) 

    # First plot
    axs[0].plot(var_list_stu_skewed_t_5, label='VaR MS CCC-GARCH Skewed-t')
    axs[0].plot([x*-1 for x in var_list_stu], label='VaR MS CCC-GARCH Stu')
    axs[0].plot([x*-1 for x in var_list_norm], label='VaR MS CCC-GARCH Norm')
    axs[0].plot([x*-1 for x in var_list_stu_single], label='CCC-GARCH Stu')
    axs[0].plot([x*-1 for x in var_list_stu_dcc], label='DCC-GARCH Stu')

    axs[0].scatter(range(len(rret_short)), rret_short, color='black')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('log returns')
    axs[0].legend()
    axs[0].legend(loc='lower right')

    axs[0].set_title('VaR 95% equally weighted short portfolio')

    # Second plot
    axs[1].plot([x*-1 for x in var_list_stu_skewed_t_95], label='VaR MS CCC-GARCH Skewed-t')
    axs[1].plot([x*-1 for x in var_list_stu], label='VaR MS CCC-GARCH Stu')
    axs[1].plot([x*-1 for x in var_list_norm], label='VaR MS CCC-GARCH Norm')
    axs[1].plot([x*-1 for x in var_list_stu_single], label='CCC-GARCH Stu')
    axs[1].plot([x*-1 for x in var_list_stu_dcc], label='DCC-GARCH Stu')

    axs[1].scatter(range(len(rret_long)), [x for x in rret_long], color='black')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('log returns')
    axs[1].legend()
    axs[1].legend(loc='lower right')

    axs[1].set_title('VaR 95% equally weighted long portfolio')

    plt.tight_layout() # Adjust subplots to fit into the figure area.
    return fig

def figure_5(df_backtest,pnl_equal,pnl_gmvp,pnl_model_skew_short,pnl_sharpe_short,
             pnl_sharpe_long,pnl_model_skew_long_short,f_end):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plotting on the first subplot
    ax1.plot(df_backtest.iloc[ - f_end :].index,np.cumsum(pnl_equal), label='Equal Weight')
    ax1.plot(df_backtest.iloc[ - f_end :].index,np.cumsum(pnl_gmvp), label='GMVP')
    ax1.plot(df_backtest.iloc[ - f_end :].index,np.cumsum(pnl_model_skew_short), label='Min CVaR',color = 'g')
    ax1.plot(df_backtest.iloc[ - f_end :].index,np.cumsum(pnl_sharpe_short), label='Max Sharpe',color = 'r')

    ax1.set_title('Portfolio Cumulative Log Returns - Short')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative Log Returns')
    ax1.legend()

    # Plotting on the second subplot
    ax2.plot(df_backtest.iloc[ - f_end :].index,np.cumsum(pnl_sharpe_long), label='Max Sharpe',color = 'r')
    ax2.plot(df_backtest.iloc[ - f_end :].index,np.cumsum(pnl_model_skew_long_short), label='Min CVaR',color = 'g')
    ax2.set_title('Portfolio Cumulative Log Returns - Long-Short')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Log Returns')
    ax2.legend()

    # Show the plot
    plt.tight_layout()
    return fig

def plot_long_short_allocation(df,title):
    """
    Plot a stacked area chart for long-short portfolio allocation over time.

    Parameters:
    - df: A pandas DataFrame with datetime index and columns for each asset's weight.
          Positive values are long positions, and negative values are short positions.
    """
    
    # Convert the index to datetime if it's not already, this line can be commented
    # if the DataFrame already has a datetime index
    df.index = pd.to_datetime(df.index)

    # Separate the DataFrame into positive and negative parts
    pos_df = df.clip(lower=0)
    neg_df = df.clip(upper=0)

    # Create the plot
    plt.figure(figsize=(10, 5))
    
    # Plot positive weights
    plt.stackplot(df.index, pos_df.T, labels=pos_df.columns, baseline='zero', colors= ['purple','lightblue','lightgreen'] )
    
    # Plot negative weights
    plt.stackplot(df.index, neg_df.T,baseline='zero', colors= ['purple','lightblue','lightgreen'])
    
    # Add a horizontal line at y=0
    plt.axhline(0, color='black', linewidth=0.5)
    
    # Customize the legend
    plt.legend(loc='upper left')
    
    # Titles and labels
    plt.title(f'{title}')
    plt.xlabel('Date')
    plt.ylabel('Commodity Weight')

    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Show plot
    plt.tight_layout()
    plt.show()