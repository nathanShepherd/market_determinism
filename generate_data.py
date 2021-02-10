import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np 

def correlate_matrix(csv_filename):
    df = pd.read_csv(csv_filename)

    df_corr = df.corr()#correlates company price data
    out_filename = csv_filename.split('.')[0] + "_corr.csv"
    df_corr.to_csv(out_filename)
    return out_filename

def heatmap_plot(X_values, X_df):
    # parameters
    save_file_as = "correlations.png"
    shape = X_values.shape + (1, 1)
    row_labels = X_df.index
    column_labels = X_df.columns
    row_idx = np.arange(shape[1]) + 0.5
    column_idx = np.arange(shape[0]) + 0.5
    
    # heatmap configuration
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    heatmap1 = ax1.pcolor(X_values, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)
    ax1.set_xticks(row_idx, minor=False)
    ax1.set_yticks(column_idx, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    ax1.set_yticklabels(row_labels)
    ax1.set_xticklabels(column_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()

    plt.savefig(save_file_as, dpi = (300))

def visualize_data():
    X = "global_idx_close_price.csv"
    filename_corr = correlate_matrix(X)
    df_corr = pd.read_csv(filename_corr)
    #import pdb; pdb.set_trace()
    data = df_corr.values
    # data = np.array of rows and columns with labels
    # data.sort(axis=0)
    # data.sort(axis=1)
    heatmap_plot(data[1:,1:], df_corr)

    #plt.show()

def save_corr(X):
    df = pd.read_csv(X)

    dates = df.iloc[:, 0]
    values = df.iloc[:,1:].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(values)

    #df_corr = df.corr()#correlates company price data
    # using instead scaled prices
    normalized_df = pd.DataFrame(x_scaled, 
                            columns=df.columns[1:],
                            index=df.index)
    if "Unnamed: 0" == df.columns[0]:
        df = df.drop(columns="Unnamed: 0")
    normalized_df.to_csv("Normal_" + X.split('/')[1])
    df_corr = normalized_df.corr()#correlates company price data
    df_corr.to_csv(X.split('.')[0] + "_corr.csv")

    data = df_corr.values#returns np.array of rows and columns
    #data.sort(axis=0)
    #data.sort(axis=1)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    plt.savefig("correlations.png", dpi = (300))

def histogram_normal(title, X = "matrix.csv"):
    # Standardize columns of X into range [Min, Max]
    # Save as png the histogram of values in Y
    # Print a breif summary of Y
    df = pd.read_csv(X).interpolate()

    print(df.iloc[:,1:].describe())
    Y = df.values[:,1:]
    Y = np.nanmean(Y, axis=1)
    #import pdb; pdb.set_trace()
    plt.hist(Y)
    plt.title(title)
    plt.savefig("normal_hist.png")
    
    #df.describe().to_csv("global_price_stats.csv")

if __name__ == "__main__":
    '''
    #visualize_data()
    
    histogram_normal("Histogram of Normalized Close Prices (G.idx)",
                    "Normal_global_idx_close.csv")
    '''
    #save_corr("joined_dfs/sp500_joined_Closed_prices.csv")
    histogram_normal("Hist. of Normal. Close Prices (SP500)",
                    "joined_dfs/Normal_sp500_joined_Closed_prices.csv")
    