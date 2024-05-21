import pandas as pd
import numpy as np
import datetime as dt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def candles_plot(input_df, subplots=False, r=1, c=2):

    plot_df = input_df.copy().reset_index()

    if subplots:

        fig = make_subplots(rows=r, cols=c, shared_xaxes=True, vertical_spacing=0.05)

        fig.add_trace(
        go.Candlestick(
            x=plot_df['time'],
            open=plot_df['mid_o'],
            high=plot_df['mid_h'],
            low=plot_df['mid_l'],
            close=plot_df['mid_c'],
            name='candles'
        ), row=1, col=1
    )

    else:
        
        fig = go.Figure()

        fig.add_trace(
        go.Candlestick(
            x=plot_df['time'],
            open=plot_df['mid_o'],
            high=plot_df['mid_h'],
            low=plot_df['mid_l'],
            close=plot_df['mid_c'],
            name='candles'
        )
        )
        
    fig.update_layout(
        height=800,
        plot_bgcolor='#fff',
        xaxis_rangeslider_visible=False
    )

    fig.update_xaxes(
        nticks=30,
        showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
        rangebreaks=[
            { 'pattern': 'day of week', 'bounds': [6, 1]},
            { 'pattern': 'hour', 'bounds':[20,12]}
        ]
    )

    fig.update_yaxes(
        nticks=30,
        tickformat="none",
        showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
        zeroline=True, zerolinewidth=1, zerolinecolor='#000',
    )

    return fig

def is_support(df, i):

    support = df['mid_l'][i] < df['mid_l'][i-1] and df['mid_l'][i] < df['mid_l'][i+1] and df['mid_l'][i+1] < df['mid_l'][i+2] and df['mid_l'][i-1] < df['mid_l'][i-2]

    return support

def is_resistance(df,i):
  
    resistance = df['mid_h'][i] > df['mid_h'][i-1] and df['mid_h'][i] > df['mid_h'][i+1] and df['mid_h'][i+1] > df['mid_h'][i+2] and df['mid_h'][i-1] > df['mid_h'][i-2]
    
    return resistance

def far_from_level(new_level, other_levels, tolerance, min_tolerance=10):

    return np.sum([abs(new_level-x) < max(tolerance, min_tolerance) for x in other_levels]) == 0


# create a level_df with columns for each support/resistance level.
# the value of the level is available only when computation is possible (so with a delay of two candles)
# an additional column for the timestamps of each level is computed
def daily_support_resistance_df(dataframe, level_dict, trading_day):

    level_df = pd.DataFrame(data=dataframe['time'])

    for i, (key,level) in enumerate(level_dict[trading_day].items()):

        level_df[f'level_{i}'] = level
        level_df[f'level_{i}_date'] = key
        level_df[f'level_{i}'] = np.where(level_df['time'] <= level_df[f'level_{i}_date'], np.nan, level_df[f'level_{i}'])
        level_df[f'level_{i}'] = level_df[f'level_{i}'].shift(2)

        level_df.drop(columns=[f'level_{i}_date'], inplace=True)

    return level_df


# create a signal dataframe in which, every time we have a support or resistance touch, we go long or short
def daily_signals_df(dataframe, level_df):

    signals_df = pd.DataFrame(data=[dataframe['time']]).T

    signals_df['buy_signal'] = 0
    signals_df['sell_signal'] = 0

    for i in list(signals_df.index)[1:]:
        
        lowest_price = dataframe.loc[i, 'mid_l']
        previous_lowest = dataframe.loc[(i-1), 'mid_l']

        highest_price = dataframe.loc[i, 'mid_h']
        previous_highest = dataframe.loc[(i-1), 'mid_h']

        levels_available = [level for level in level_df.iloc[i][1:] if level > 0]

        # buy signals

        support_distances = [abs(level-lowest_price) for level in levels_available]

        # by using zip I link the available levels with their distances from the current lowest price
        # I then pick the support which is closest to the current price (by selecting the one with minimum distance)
        support = min(zip(levels_available,support_distances), key=lambda x: x[1])[0] if len(list(zip(levels_available,support_distances))) > 0 else None

        if support != None and previous_lowest > support and lowest_price < support:

            signals_df.loc[i, 'buy_signal'] = 1
            signals_df.loc[i, 'signal_price'] = support
            signals_df.loc[i, 'low_price'] = lowest_price

        # sell signals

        resistance_distances = [abs(level-highest_price) for level in levels_available]
        resistance = min(zip(levels_available,resistance_distances), key=lambda x: x[1])[0] if len(list(zip(levels_available,resistance_distances))) > 0 else None

        if resistance != None and previous_highest < resistance and highest_price > resistance:

            signals_df.loc[i, 'sell_signal'] = -1
            signals_df.loc[i, 'signal_price'] = resistance
            signals_df.loc[i, 'high_price'] = highest_price

    return signals_df


# now that I have a signal df, I can create a strategy that considers opening and closing positions related to the signals
# let's create a first easy one. The easy one applies the following idea:
# it opens the positions following the buy and sell signals with fixed stop loss and take profit 
# it flips the switch in case of a contrarian signal, closing all the current positions.
def strat_1(day_df, signals_df, slippage=1, tp=10, sl=10):

    day_df = day_df.reset_index(drop=True)
    
    strat_df = day_df[['time']].copy()

    trades = pd.DataFrame()

    # keep track of open positions (1 if long, -1 if short)
    position = 0
    stop_loss = 0
    take_profit = 0
    
    order_id = 0
    
    # we eiterate for each candle/minute available
    for i in day_df.index:

        current_minute = day_df.loc[i,'time']

        buy_signal = signals_df.loc[i, 'buy_signal']
        sell_signal = signals_df.loc[i, 'sell_signal']
        
        # if eod and position opened
        if day_df.iloc[i]['time'] == day_df.iloc[-1]['time']:
            if position == 0:
                continue
            elif position == 1:
                position = 0
                strat_df.loc[i, 'realized_profit'] = day_df.loc[i, 'bid_o'] - entry_price
                strat_df.loc[i, 'buy_order'] = -1
                trades.loc[trades.trade_id == order_id, 'exit_time'] = current_minute
                trades.loc[trades.trade_id == order_id, 'closed_price'] = day_df.loc[i, 'bid_h']
                trades.loc[trades.trade_id == order_id, 'realized_profit'] = day_df.loc[i, 'bid_h'] - entry_price
            else:
                position = 0
                strat_df.loc[i, 'realized_profit'] = entry_price - day_df.loc[i, 'ask_o']
                strat_df.loc[i, 'buy_order'] = -1
                trades.loc[trades.trade_id == order_id, 'exit_time'] = current_minute
                trades.loc[trades.trade_id == order_id, 'closed_price'] = day_df.loc[i, 'ask_o']
                trades.loc[trades.trade_id == order_id, 'realized_profit'] = entry_price - day_df.loc[i, 'ask_o']


        # case of buy signal
        if buy_signal == 1:

            # case of no open positions
            if position == 0:

                position = 1
                order_id +=1
                entry_price = signals_df.loc[i, 'signal_price'] + slippage

                trades.loc[i, 'trade_id'] = order_id
                trades.loc[i, 'type'] = position
                trades.loc[i,'open_time'] = current_minute
                trades.loc[i,'entry_price'] = entry_price

                strat_df.loc[i, 'buy_order'] = 1
                strat_df.loc[i, 'open_price'] = entry_price
                stop_loss = entry_price - sl
                take_profit = entry_price + tp

                strat_df.loc[i, 'stop_loss'] = stop_loss
                strat_df.loc[i, 'take_profit'] = take_profit

            # case already long (do not do anything, price going down basically)
            elif position == 1:
                continue

            elif position == -1:

                # we flip the switch
                strat_df.loc[i, 'realized_profit'] = min(tp, entry_price - signals_df.loc[i, 'signal_price'])
                strat_df.loc[i, 'sell_order'] = -1

                trades.loc[trades.trade_id == order_id, 'exit_time'] = current_minute
                trades.loc[trades.trade_id == order_id, 'closed_price'] = max(take_profit, signals_df.loc[i, 'signal_price'])
                trades.loc[trades.trade_id == order_id, 'realized_profit'] = min(tp, entry_price - signals_df.loc[i, 'signal_price'])

                position = 1
                order_id +=1
                entry_price = signals_df.loc[i, 'signal_price'] + slippage

                trades.loc[i, 'trade_id'] = order_id
                trades.loc[i, 'type'] = position
                trades.loc[i,'open_time'] = current_minute
                trades.loc[i,'entry_price'] = entry_price

                strat_df.loc[i, 'buy_order'] = 1
                strat_df.loc[i, 'open_price'] = entry_price
                stop_loss = entry_price - sl
                take_profit = entry_price + tp

                strat_df.loc[i, 'stop_loss'] = stop_loss
                strat_df.loc[i, 'take_profit'] = take_profit

        # case of sell signal
        elif sell_signal == -1:
            
            # sell short if no positions open
            if position == 0:

                position = -1
                order_id +=1
                entry_price = signals_df.loc[i, 'signal_price'] - slippage

                trades.loc[i,'trade_id'] = order_id
                trades.loc[i,'type'] = position
                trades.loc[i,'open_time'] = current_minute
                trades.loc[i,'entry_price'] = entry_price

                strat_df.loc[i, 'sell_order'] = 1
                strat_df.loc[i, 'open_price'] = entry_price
                stop_loss = entry_price + sl
                take_profit = entry_price - tp

                strat_df.loc[i, 'stop_loss'] = stop_loss
                strat_df.loc[i, 'take_profit'] = take_profit

            # case already short (do not do anything, price going up basically)
            elif position == -1:
                continue
            
            # case we are long
            elif position == 1:

                # we flip the switch
                strat_df.loc[i, 'realized_profit'] = min(tp, signals_df.loc[i, 'signal_price'] - entry_price)
                strat_df.loc[i, 'buy_order'] = -1

                trades.loc[trades.trade_id == order_id, 'exit_time'] = current_minute
                trades.loc[trades.trade_id == order_id, 'closed_price'] = min(take_profit, signals_df.loc[i, 'signal_price'])
                trades.loc[trades.trade_id == order_id, 'realized_profit'] = min(tp, signals_df.loc[i, 'signal_price'] - entry_price)

                position = -1
                order_id +=1
                entry_price = signals_df.loc[i, 'signal_price'] - slippage

                trades.loc[i,'trade_id'] = order_id
                trades.loc[i,'type'] = position
                trades.loc[i,'open_time'] = current_minute
                trades.loc[i,'entry_price'] = entry_price

                strat_df.loc[i, 'sell_order'] = 1
                strat_df.loc[i, 'open_price'] = entry_price
                stop_loss = entry_price + sl
                take_profit = entry_price - tp

                strat_df.loc[i, 'stop_loss'] = stop_loss
                strat_df.loc[i, 'take_profit'] = take_profit
        
        
        # case of no signals at current time
        else:
            
            # case we are already long
            if position == 1:

                high = day_df.loc[i, 'bid_h']
                low = day_df.loc[i, 'ask_l']
                
                # check if we are stopped-out
                if low <= stop_loss:

                    position = 0
                    strat_df.loc[i, 'realized_profit'] = stop_loss - entry_price
                    strat_df.loc[i, 'buy_order'] = -1

                    trades.loc[trades.trade_id == order_id, 'exit_time'] = current_minute
                    trades.loc[trades.trade_id == order_id, 'closed_price'] = stop_loss
                    trades.loc[trades.trade_id == order_id, 'realized_profit'] = stop_loss - entry_price


                # check if we took profit
                elif high >= take_profit:

                    position = 0

                    strat_df.loc[i, 'realized_profit'] = take_profit - entry_price
                    strat_df.loc[i, 'buy_order'] = -1

                    trades.loc[trades.trade_id == order_id, 'exit_time'] = current_minute
                    trades.loc[trades.trade_id == order_id, 'closed_price'] = take_profit
                    trades.loc[trades.trade_id == order_id, 'realized_profit'] = take_profit - entry_price
                    

            # case we are already short
            if position == -1:

                high = day_df.loc[i, 'ask_h']
                low = day_df.loc[i, 'bid_l']
                
                # check if we are stopped-out
                if high >= stop_loss:

                    position = 0
                    strat_df.loc[i, 'realized_profit'] = entry_price - stop_loss
                    strat_df.loc[i, 'sell_order'] = -1

                    trades.loc[trades.trade_id == order_id, 'exit_time'] = current_minute
                    trades.loc[trades.trade_id == order_id, 'closed_price'] = stop_loss
                    trades.loc[trades.trade_id == order_id, 'realized_profit'] = entry_price - stop_loss

                # check if we took profit
                elif low <= take_profit:

                    position = 0
                    strat_df.loc[i, 'realized_profit'] = entry_price - take_profit
                    strat_df.loc[i, 'sell_order'] = -1

                    trades.loc[trades.trade_id == order_id, 'exit_time'] = current_minute
                    trades.loc[trades.trade_id == order_id, 'closed_price'] = take_profit
                    trades.loc[trades.trade_id == order_id, 'realized_profit'] = entry_price - take_profit

            else:
                continue
                
    strat_df.fillna(0, inplace=True)

    return strat_df, trades

########################################### init df #######################################
# prepare the initial_df
initial_df = pd.read_pickle('data/NAS100_M1_2021_2022.pickle')
initial_df['time'] = pd.to_datetime(initial_df['time'])
initial_df.head()


# prepare the df_2122
df_2122 = initial_df[initial_df['time'].dt.year >= 2021].reset_index(drop=True)
df_2122 = df_2122[(df_2122['time'].dt.time >= dt.time(12,0,0)) & (df_2122['time'].dt.time <= dt.time(19,0,0))].reset_index(drop=True)

# compute the range of the candles for df_2122
# compute the 5-period rolling mean of the range. It is used for giving a tolerance range to the support/resistance levels
df_2122['mid_range'] = df_2122['mid_h'] - df_2122['mid_l']
df_2122['mid_range_5_rolling_mean'] = df_2122['mid_range'].rolling(5).mean()



# create a dictionary of support and resistance levels
# the levels are checked every minute
# the levels are inserted into the dictionary if they pass the 'far_from_level' screening ( reason is we don't want levels too close )
levels = dict()

for unique_day in df_2122['time'].dt.date.unique():

  levels[unique_day] = dict()

  temp_df = df_2122[df_2122['time'].dt.date == unique_day].reset_index()

  for i in range(2,temp_df.shape[0]-2):
    
    if is_support(temp_df,i):

      support = temp_df['mid_l'][i]

      if far_from_level(support, levels[unique_day].values(), temp_df['mid_range_5_rolling_mean'][i]):

        levels[unique_day][temp_df['time'][i]] = support
      
    elif is_resistance(df_2122,i):

      resistance = temp_df['mid_h'][i]

      if far_from_level(resistance, levels[unique_day].values(), temp_df['mid_range_5_rolling_mean'][i]):

        levels[unique_day][temp_df['time'][i]] = resistance


########################################### main body #######################################
strat_1_df = pd.DataFrame()
trades_strat_1 = pd.DataFrame()

for day in list(df_2122['time'].dt.date.unique()):

    if day.year == 2021:

        day_df = df_2122[df_2122['time'].dt.date == day].copy()
        day_df.reset_index(inplace=True, drop=True)

        level_df = daily_support_resistance_df(day_df, levels, day)
        signals_df = daily_signals_df(day_df, level_df)
        strat_df, trades_df = strat_1(day_df, signals_df, slippage=0.5, tp=75, sl=25)

        strat_1_df = pd.concat([strat_1_df,strat_df])
        trades_strat_1 = pd.concat([trades_strat_1,trades_df])

strat_1_df['rolling_profit'] = strat_1_df['realized_profit'].cumsum()
plot_df = pd.merge(strat_1_df, df_2122, how='inner')
fig = candles_plot(plot_df, subplots=True, r=2, c=1)

# Rolling profits
fig.add_trace(
    go.Scatter(
    x = plot_df[f'time'],
    y = plot_df['rolling_profit'],
    marker_color='#0099e6',
    name='rolling profit'
    ), row=2, col=1
)

fig.show()
