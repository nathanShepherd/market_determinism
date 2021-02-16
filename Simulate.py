import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 

plt.style.use("ggplot")
np.random.seed(406)

def simulate_Sn(nSamples, matrix):
    # select subset of columns from a sample
    rows, cols = matrix.shape
    col_sample_idx = np.random.choice(np.arange(0, cols), nSamples)
    return (matrix.T[col_sample_idx]).T

def volatility(df):
    # Compute coef of variance to determine volatility
    coef_var = df.std() / df.mean()
    return coef_var

def recent_concavity(X, t, h):
    # Returns whether price trend is up or down
    degree = 2
    num_days_past = t - h
    axis = np.arange(num_days_past, t)
    price = np.mean(X[num_days_past:t], axis=1)
    coef_fit = np.polyfit(axis, price, degree)

    # If regressor for x^2 is positive
    # price change has upward concavity
    return coef_fit[0]

def simulate_trading_return(filename):
    prices = pd.read_csv(filename)
    prices = prices.dropna(axis=1).interpolate()
    prices = prices.iloc[:,2:].values
    start = 1000
    Sn_size = 100000 # Num of stock to simulate
    Sn = simulate_Sn(Sn_size, prices)
    per_inv = start / Sn_size

    intervals = [30, 90, 365, 365*2,] # days
    terminals = []
    for h in intervals:
        exp_equity = [start]
        buy_idx = 1
        for t in range(h + 1, len(prices[:,0])):
            curr_eq = per_inv * ((Sn[t] / Sn[t-h]) - 1)
            '''
            coef_var = recent_concavity(Sn, t, h)
            if coef_var > 0 :
                # Sell given positive d/dP
                buy_idx = h
            elif coef_var < 0:
                # Buy given negative d/dP
                buy_idx = t
            '''
            exp_equity.append(sum(curr_eq))

        # Simulate random samples of equity
        exp_equity = np.random.choice(exp_equity, 10000, replace=True)
        terminals.append(exp_equity)

    h_equity = dict(zip(intervals,terminals))
    pd.DataFrame(h_equity).plot(kind="density")
    title = f"Simulated Sn={Sn_size} E[equity] over n days"
    plt.title(title)
    plt.xlabel("E[equity]")
    plt.xlim([-1000,1000])
    plt.savefig(title)

def simulate_equity(filename):
    P = pd.read_csv(filename)
    P = P.dropna(axis=1).interpolate()
    #
    start = 1000    
    Sn = P.iloc[:,2:].values
    per_inv = start / len(P.iloc[1])
    rows, cols = Sn.shape
    X = Sn
    
    #num_Sn_samples = 1000
    #col_sample_idx = np.random.choice(np.arange(0, cols), num_Sn_samples)
    #X = (X.T[col_sample_idx]).T

    # equity for $1000 over h days
    # equity = $ invested times the diff of buy to sell price
    # eq = (start / Price(t)) * (P(t+h) - P(t))
    #    = start * ( [P(t+h) / P(t)] - 1)
    
    
    intervals = [7,90,365,356*2] # days
    terminals = []
    for h in intervals:
        exp_equity = [start]
        for t in range(h + 1, rows):
            curr_eq = per_inv * ((X[t] / X[t-h]) - 1)
            exp_equity.append(sum(curr_eq))

        # Simulate random samples of equity
        exp_equity = np.random.choice(exp_equity, 10000, replace=True)
        terminals.append(exp_equity)

    h_equity = dict(zip(intervals,terminals))

    #import pdb; pdb.set_trace()

    pd.DataFrame(h_equity).plot(kind="density")
    #plt.hist(exp_equity, density=True)
    title = "Simulated E[equity] over n day intervals"
    plt.title(title)
    plt.xlabel("E[equity]")
    plt.savefig(title)

    pd.DataFrame(h_equity).to_csv("R/exp_return_n_days")



if __name__ == "__main__":
    #simulate("joined_dfs/sp500_joined_Closed_prices_interp.csv")
    simulate_trading_return("joined_dfs/sp500_joined_Closed_prices_interp.csv")
