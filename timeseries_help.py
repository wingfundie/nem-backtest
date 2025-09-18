import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy import pad
import math
import statsmodels
import datetime
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import polyfit
# import lightgbm as lgb
# from lightgbm import LGBMRegressor
import itertools

import phik
from phik.report import plot_correlation_matrix
from phik import report

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import grangercausalitytests 
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
from scipy.fftpack import fft

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report,f1_score, roc_curve, roc_auc_score, auc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# graphing libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## GENERAL HELPERS ##

# Rolling Spearman Correlation
def rolling_spearman(seqa, seqb, window):
    seqa = np.array(seqa)
    seqb = np.array(seqb)
    stridea = seqa.strides[0]
    ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
    strideb = seqa.strides[0]
    ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    ar = ar.rank(1)
    br = br.rank(1)
    corrs = ar.corrwith(br, 1)
    return pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)


# Rolling Correlation Generator 
def gen_rolling_corr (df, rolling_frame, target, pearson = True):
    rolling_df = df.copy()

    # Loop through columns and calculate rolling correlations
    for col in df.columns:
        #if col == date_col:
        #    pass
        if not pearson:
            rolling_df[f'{col}'] = rolling_spearman(df[target], df[col], rolling_frame)
        else:
            rolling_df[f'{col}'] = df[target].rolling(window=rolling_frame).corr(df[col])
    return rolling_df
"""
def gen_rolling_corr (df, rolling_frame, target, date_col):
    rolling_df = df.loc[:, [date_col]]

    # Loop through columns and calculate rolling correlations
    for col in df.columns[1:]:
        if col == date_col:
            pass  
        rolling_df[f'{col}'] = df[target].rolling(window=rolling_frame).corr(df[col])
    return rolling_df
"""

# PHIK Corr Matrix
def phik_matrix (data):
    #Using Phi-K Coefficient to determine variables with highest correlation to Gasoline Inventories
    df = data
    interval_cols= list(df.columns)
    phik_overview = df.phik_matrix(interval_cols= interval_cols)

    #compare= phik_overview.loc[:,"us_gasoline_inv"]

    plot_correlation_matrix(phik_overview.values, 
                            x_labels=phik_overview.columns, 
                            y_labels=phik_overview.index, 
                            vmin=0, vmax=1, color_map="Blues", 
                            title=r"correlation $\phi_K$", 
                            fontsize_factor=0.8, 
                            figsize=(10, 8))
    plt.tight_layout()
    return phik_overview



# Simple Linear Regression
def linear_regress (data_fill, X, y, multi = False):
    model = LinearRegression()
    if multi:
        X = data_fill[X]
        y = data_fill[y]

        # Create and fit the linear regression model
        
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # MULTIVARIATE
        # Calculate the R-squared value
        r_squared = model.score(X, y)

        # Print the R-squared value
        print(f"R-squared: {r_squared}")

        # Plot the actual data points
        plt.scatter(y, y_pred, c='blue', label='Actual vs. Predicted')

        # Plot the best-fit line
        plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', c='red', linewidth=2, label='Best Fit Line')

        # Set labels and legend
        plt.xlabel('Spreads')
        plt.ylabel('PX Timespread')
        plt.legend()


        return model

    # Select the feature columns and target column
    X = data_fill[X].values.reshape(-1, 1)
    y = data_fill[y].values
    
    # Create and fit the linear regression model
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate R-squared
    r2 = r2_score(y, y_pred)
    print(f'R-squared: {r2:.4f}')

    # Plot the data points and best fit line
    plt.scatter(data_fill[X], data_fill[y], label='Data Points', color='blue')
    plt.plot(data_fill[X], y_pred, label='Best Fit Line', color='red')
    plt.xlabel('Spreads')
    plt.ylabel('PX Timespread')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    return model

# OLS Linear Regression
def linear_model(data, X_names, y_name, residual_plot=False):
    # Ensure X_names is a list
    if isinstance(X_names, str):
        X_names = [X_names]
    
    # Select the independent variables and add a constant
    X_er = data[X_names]
    X = sm.add_constant(X_er)
    y = data[y_name]

    # Fit the model
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Plotting only the first predictor against y, as we can't plot multivariate regression in 2D
    if len(X_names) == 1:
        intercept = model.params[0]
        slope = model.params[1]

        # Calculate predicted y values based on the best-fit line
        x_pred = np.linspace(min(X_er[X_names[0]]), max(X_er[X_names[0]]), len(y))
        y_pred = intercept + slope * x_pred

        r2 = model.rsquared

        # Plot the data and the best-fit line
        plt.scatter(X_er[X_names[0]], y)
        plt.plot(x_pred, y_pred, color='red')

        # Add labels and title
        plt.xlabel(f'{X_names[0]}')
        plt.ylabel(f'{y_name}')
        plt.title('Best Fit Line')

        equation = "y = {:.5f} + {:.5f}x".format(intercept, slope)
        # Add a text annotation for R-squared value
        plt.annotate("R-squared: {:.3f}".format(r2), (0.05, 0.8), xycoords='axes fraction', ha='left')

        # Add a text annotation for the equation of the best-fit line
        plt.annotate(equation, (0.05, 0.9), xycoords='axes fraction', ha='left')

        # Show the plot
        plt.show()

    if residual_plot:
        # Calculate residuals
        residuals = y - model.predict(X)

        # Plot the residuals
        plt.plot(residuals)
        plt.xlabel('Index')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

        plot_acf(residuals, lags=15)
        plt.show()

    return model

def create_residuals(price_df, Y_var, x_var):
    """
    Calculate the OLS and create the beta hedge ratio and residuals for the two assets

    Parameters
    ----------
    price_df : `pd.DataFrame`
        A DataFrame containing asset data  from different dates. 
        Index is a Datetime object.

    Returns
    -------
    price_df : `pd.DataFrame`
        Updated DataFrame with column values for beta hedge ratio (beta_hr) and 
        residuals (Residuals).
    """
    # Create OLS model
    Y = price_df[Y_var]
    x = price_df[x_var]
    x = sm.add_constant(x)
    model = sm.OLS(Y, x)
    res = model.fit()
    
    # Beta hedge ratio (coefficent from OLS)
    beta_hr = res.params[1]
    print(f'Beta Hedge Ratio: {beta_hr}')
    
    # Residuals
    price_df["residuals"] = res.resid
    return price_df

# Memory usage reduction
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        try:
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max <                  np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')
        except Exception as e:
            print(f'ERROR FOR {col}')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# Outlier Checking
def outlier_thresholds(dataframe, col_name, q1_perc=0.05, q3_perc=0.95):
    """
    given dataframe, column name, q1_percentage and q3 percentage, function calculates low_limit and up_limit

    """
    quartile1 = dataframe[col_name].quantile(q1_perc)
    quartile3 = dataframe[col_name].quantile(q3_perc)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1_perc=0.01, q3_perc=0.99):
    outlier_list = []
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1_perc=0.01, q3_perc=0.99)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True

    else:
        return False

# Calculate summary 
def calculate_summary_statistics(data, target_column):
    """
    Calculate summary statistics for the target sentiment labels.
    Args:
    data (pd.DataFrame): Input dataset as a pandas DataFrame.
    target_column (str): Column name of the target sentiment labels.

    Returns:
    summary_statistics (pd.DataFrame): Summary statistics as a pandas DataFrame.
    """
    summary_statistics = data[target_column].describe()
    return summary_statistics

# Summary Stats
def stats (df):
    skewness = df.skew()
    kurtosis = df.kurtosis()
    #df.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    # Display the calculated statistics
    print("----- SUMMARY STATISTICS -----")
    print(f"Mean: {df.mean():.5f}")
    print(f"Standard Deviation: {np.std(df):.5f}")
    print(f"Max: {max(df):.5f}")
    print(f"Min: {min(df):.5f}")
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")
    print(f"5%: {df.quantile(0.05):.6f}")
    print(f"25%: {df.quantile(0.25):.6f}")
    print(f"50%: {df.quantile(0.50):.6f}")
    print(f"75%: {df.quantile(0.75):.6f}")
    print(f"95%: {df.quantile(0.95):.6f}")

# Summary Stats DF
def summary_stats_df(dfs, df_headers):
    summary_df = pd.DataFrame()
    for i in range(len(dfs)):
        summary_df.loc['Mean', df_headers[i]] = f"{dfs[i].mean():.5f}"
        summary_df.loc['SD', df_headers[i]] = f"{np.std(dfs[i]):.5f}"
        summary_df.loc['Max', df_headers[i]] = f"{max(dfs[i]):.5f}"
        summary_df.loc['Min', df_headers[i]] = f"{min(dfs[i]):.5f}"
        summary_df.loc['Skew', df_headers[i]] = f"{dfs[i].skew():.5f}"
        summary_df.loc['Kurt', df_headers[i]] = f"{dfs[i].kurtosis():.5f}"
        summary_df.loc['5%', df_headers[i]] = f"{dfs[i].quantile(0.05):.5f}"
        summary_df.loc['25%', df_headers[i]] = f"{dfs[i].quantile(0.25):.5f}"
        summary_df.loc['50%', df_headers[i]] = f"{dfs[i].quantile(0.50):.5f}"
        summary_df.loc['75%', df_headers[i]] = f"{dfs[i].quantile(0.75):.5f}"
        summary_df.loc['95%', df_headers[i]] = f"{dfs[i].quantile(0.95):.5f}"
        
    return summary_df

## TIMESERIES DECOMPOSITION ##

## Autocorrelation Plots
def tsplot(y, name, lags=None, figsize=(12, 7)):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
           
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    p_value = sm.tsa.stattools.adfuller(y)[1]
    ts_ax.set_title(f'Time Series Analysis Plots ({name})\n Dickey-Fuller: p={0:.5f}'.format(p_value))
    plot_acf(y, lags=lags, ax=acf_ax)
    plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()

## DETRENDING
# Regression Detrending
def regression_detrend (data, col, degree = 1):
    col = col
    if degree <2:
        # fit linear model
        X = [i for i in range(0, len(data[col]))]
        X = np.reshape(X, (len(X), 1))
        y = data[col].values

        model = LinearRegression()
        model.fit(X, y)
        # calculate trend
        trend = model.predict(X)
        trend_frame = pd.DataFrame(trend, columns= ['trend'])

        # plot trend
        plt.plot(y)
        plt.plot(trend)
        plt.show()

        # detrend
        detrender = list(data[col] - trend_frame['trend'])
        detrender_frame = pd.DataFrame(detrender, columns= [f'detrend_{col}'])
        # plot detrended
        # pyplot.plot(detrender)
        # pyplot.show()
    else:
        degree = degree
        # Extract the column data
        ser = data[col]

        # Non-linear trend
        X = ser.index.values
        y = ser.values

        pf = PolynomialFeatures(degree=degree)
        Xp = pf.fit_transform(X.reshape(-1, 1))
        md2 = LinearRegression()
        md2.fit(Xp, y)
        trend = md2.predict(Xp)
        trend_frame = pd.DataFrame(trend, columns= ['trend'])

        plt.plot(X, y)
        plt.plot(X, trend)
        plt.legend(['data', 'polynomial trend'])
        plt.show()

    # detrend
    detrender = list(data[col] - trend_frame['trend'])
    detrender_frame = pd.DataFrame(detrender, columns= [f'detrend_{col}']) 
    plt.plot(detrender)                                              

    return detrender_frame

# Seasonality Removal

# Seasonality variable creation
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    # 1.1.2013 is Tuesday, so our starting point is the 2nd day of week
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df


def deseasonalize(data, cols, period = 252):
    # Assuming trading year there are 252 days
    deszn_spreads = data.loc[:, ['date']]
    for prod in cols:
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data[prod], period = period)
        decomp = decomposition.seasonal
        
        # Create a DataFrame for the seasonal component
        seasonal_frame = pd.DataFrame({'seasonality': decomp})
        #seasonal_frame.plot()
        
        deszn_spreads[f'ds_{prod}'] = data[prod] - seasonal_frame['seasonality']
    return deszn_spreads

# Not really in use but a seasonal decomposition as well
def seasonal_decomp(data, column):
    column = column
    X = [i%365 for i in range(0, len(data[column]))]
    y = data[column].values

    degree = 5
    coef = polyfit(X, y, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value)
    # plot curve over original data
    plt.plot(data[column].values)
    plt.plot(curve, color='red', linewidth=3)
    plt.show()

# Seasonal differencing with known periods
def seasonal_diff(df, col, period=52):
  """Performs seasonal differencing on a DataFrame, with the additional condition that no differencing is performed on rows with an index less than 52.

  Args:
    df: A Pandas DataFrame containing the time series data.
    period: The seasonal period.

  Returns:
    A Pandas DataFrame containing the seasonally differenced time series data.
  """

  seasonally_differenced_df = df.loc[:, ['date', f'{col}']]

  # Only perform seasonal differencing on rows with an index greater than or equal to 52.
  seasonally_differenced_df.loc[52:, [f'{col}']] -= df.loc[:, [f'{col}']].shift(period)
  seasonally_differenced_df.loc[:52 , [f'{col}']] = seasonally_differenced_df.loc[:52 , [f'{col}']].diff()

  return seasonally_differenced_df



from scipy.fftpack import fft
# Fourier Transform
def get_fourier(data, column):
    ffter = fft((data[column] - data[column].mean()).values)
    plt.plot(1./13 * np.arange(len(ffter)), np.abs(ffter))
    plt.title(f"FFT of {column}")
    #plt.ylim([0,120000])
    plt.xlim([365 -30, 365 + 30])
    plt.xlabel('Frequency ($y^{-1}$)')


## STATISTICAL TESTS

# Granger Causation Test
def grangers_causation_matrix(data, variables, maxlag = 10, test='ssr_chi2test', verbose = True):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

# Johanson Cointegration Test
def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

# Augmented Dickey Fuller Test
def adf (data):

    # Testing for stationarity of sentiment scores
    if type(data) == list:
        for entry in data:
                #print(data.columns[2])
                adf = adfuller(entry)
                print('ADF Statistic: %f' % adf[0])
                print('p-value: %f' % adf[1])
                print('Critical Values:')
                for key, value in adf[4].items():
                    print('\t%s: %.3f' % (key, value))
                print('--------------------')
    else:
        #print(data.columns[2])
        adf = adfuller(data)
        print('ADF Statistic: %f' % adf[0])
        print('p-value: %f' % adf[1])
        print('Critical Values:')
        for key, value in adf[4].items():
            print('\t%s: %.3f' % (key, value))

# ADF Test with more descriptions
def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':
            round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): 
        return str(val).ljust(length)
    
    print(f'Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f'Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f'Significance Level = {signif}')
    print(f'Test Statistic = {output["test_statistic"]}')
    print(f'No. Lags Chosen = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')
        
        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary.")
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary.")



def create_cadf(price_df):
    """
    Calculate the Cointegrated Augmented Dickey Fuller test on the residuals.

    Parameters
    ----------
    price_df : `pd.DataFrame`
        Updated DataFrame with column values for beta hedge ratio (beta_hr) and 
        residuals (Residuals).
    
    Returns
    -------
    cadf : `tuple`
        Results of ADF test on residuals including the test statistic,
        pvalue and critical values.
    """
    cadf = adfuller(price_df["residuals"])
    print(f'CADF:{cadf}')
    return cadf

# Hurst Exponent
def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    hurst = reg[0]

    if hurst <0.5:
        statement = f"HURST EXPONENT: {hurst} ; SERIES IS MEAN-REVERTING"
    if hurst == 0.5:
        statement = f"HURST EXPONENT: {hurst} ; SERIES IS A RANDOM WALK"
    if hurst > 0.5:
        statement = f"HURST EXPONENT: {hurst} ; SERIES IS TRENDING"

    print(statement)
    #return hurst


## FORECASTING ##

# Reverese the detrending
def reverse_regression_detrend(detrended_data, trend_data, degree=1):
    if degree < 2:
        # If the original detrending was done with a linear model
        trend = trend_data['trend'].values
    else:
        # If the original detrending was done with a polynomial model
        X = detrended_data.index.values
        Xp = PolynomialFeatures(degree=degree).fit_transform(X.reshape(-1, 1))
        trend = LinearRegression().fit(Xp, detrended_data.values).predict(Xp)

    # Add the trend back to the detrended data
    original_data = pd.DataFrame(detrended_data.values + trend, columns=[detrended_data.columns[0]])

    # Plot the original data with the added trend
    plt.plot(original_data)
    plt.plot(trend_data['trend'])
    plt.legend(['original data', 'trend'])
    plt.show()

    return original_data

# Reverese a seasonal differencing operation
def reverse_differencing(data, col, period=52, seasonal=False):
    reversed_data = pd.DataFrame(columns=[col], index=data.index)
    
    if seasonal:
        # Seasonal differencing
        reversed_data.iloc[:period, 0] = data.iloc[:period, 0]
        for i in range(period, len(reversed_data)):
            reversed_data.iloc[i, 0] = data.iloc[i, 0] + reversed_data.iloc[i - period, 0]
    else:
        # Regular (non-seasonal) differencing
        reversed_data.iloc[0, 0] = data.iloc[0, 0]
        for i in range(1, len(reversed_data)):
            reversed_data.iloc[i, 0] = data.iloc[i, 0] + reversed_data.iloc[i - 1, 0]
    
    return reversed_data

# VAR #

# Generate combinations of variables for VAR
def gen_combinations (list_of_vars, verbose = True, exclusion = False, exclude_var = None):
    combinations = []
    for r in range(2, len(list_of_vars)):  # Generate combinations for 2 to n elements
        combos = itertools.combinations(list_of_vars, r)
        combinations.extend(combos)

    if exclusion:
        # Filter combinations to exclude those without 'dsdt_us_us_gasoline_inv'
        combinations = [list(combo) for combo in combinations if exclude_var in combo]

        # Filter combinations to exclude those where both 'us_gasoline_inv' and 'dsdt_us_us_gasoline_inv' are together
        #combinations = [combo for combo in combinations if not ('us_gasoline_inv' in combo and 'dsdt_us_gasoline_inv' in combo)]
    else:
        combinations = [list(x) for x in combinations]

    if verbose:
        # Print the unique combinations
        for i, combo in enumerate(combinations, start=1):
            print(f'Combination {i}: {combo}')

    return combinations

# Generate VAR lags based on AIC Criteria
def generate_var_lags(data, combinations, max_lags):

    best_lags_for_combinations = {}

    best_aic = float('inf')
    best_bic = float('inf')
    best_model_aic = None
    best_model_bic = None
    best_feature_combination_aic = None
    best_feature_combination_bic = None

    # Grid search for lag orders and feature combinations
    for features in combinations:
        best_aic = float('inf')
        best_model_aic = None

        # Create a new DataFrame with selected features
        selected_data = data.dropna().loc[:, features]

        # Loop through different lag orders
        for lag in range(1, max_lags):  # You can adjust the lag range as needed
            model = VAR(selected_data)
            results = model.fit(lag)

            aic_val = results.aic

            if aic_val < best_aic:
                best_aic = aic_val
                best_model_aic = results

        best_lags_for_combinations[str(features)] = best_model_aic.k_ar
    return best_lags_for_combinations


# Lightgbm

# Runs lightgbm on selected data
def lgbm_runner (data, targets, params= False, l1_reg = False, l2_reg = False, verbose = True, regression = True, early_stop = False):
    """
    data: Data in the form of a dataframe
    targets: List of target variables
    params: Dictionary containing the different parameters for feeding into lighgbm

    """
    errors = []
    for imps in targets:
        
        train = data.iloc[:int(len(data)*0.8), :]
        test = data.iloc[int(len(data)*0.8):, :]

        cols = [col for col in train.columns if col not in ['date', imps]]

        Y_train = train[imps]
        X_train = train[cols]

        Y_val = test[imps]
        X_val = test[cols]

        if regression:
            obj = 'regression'
            metric = 'mae'
        else:
            obj = 'binary'
            metric = 'auc'

        if params:
            params = params
        else:
            params = {'max_depth': 200, 
                      'num_leaves': 400,
                      'objective': obj,
                      #'early_stopping_round': 50, 
                      'metric': metric} 
            if early_stop:
                    params['early_stopping_round'] = 50

        lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
        lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

        # With L1 Regularisation
        # Should use cross-validation not just pure iteration
        if l1_reg:
            best_lambda_l1 = None
            best_error = float('inf')

            lambda_l1_values = [x * 0.1 for x in range(0, 101)]

            for lambda_l1 in lambda_l1_values:
                
                params = {
                    'max_depth': 100,
                    'num_leaves': 200,
                    'objective': obj,
                    'metric': metric,
                    'lambda_l1': lambda_l1

                }

                if early_stop:
                    params['early_stopping_round'] = 50
                
                # Perform cross-validation
                cv_results = lgb.cv(params, lgbtrain, num_boost_round=200, stratified=False, nfold=10)
                
                if regression:
                # Get the best MAE
                    mean_mae = min(cv_results['valid l1-mean'])
                    if mean_mae < best_error:
                        best_error = mean_mae
                        best_lambda_l1 = lambda_l1

                    errors.append(f"Lambda L1: {lambda_l1}, MAE: {mean_mae}")

                else:
                    mean_auc = max(cv_results['valid auc-mean'])
                    if mean_auc > best_error:
                        best_error = mean_auc
                        best_lambda_l2 = lambda_l2

            # Train the model with the best lambda_l1
            params['lambda_l1'] = best_lambda_l1
            bst_lgb = lgb.train(params, lgbtrain, 200, valid_sets=[lgbval])

        # With L2 Regularisation
        elif l2_reg:
            best_lambda_l2 = None
            best_error = float('inf')

            lambda_l2_values = [x * 0.1 for x in range(0, 101)]

            for lambda_l2 in lambda_l2_values:
                
                params = {
                    'max_depth': 100,
                    'num_leaves': 200,
                    'objective': obj,
                    'metric': metric,
                    'lambda_l2': lambda_l2

                }

                if early_stop:
                    params['early_stopping_round'] = 50
                
                # Perform cross-validation
                cv_results = lgb.cv(params, lgbtrain, num_boost_round=200, stratified=False, nfold=10)
                
                if regression:
                    # Get the best MAE
                    mean_mae = min(cv_results['valid l1-mean'])
                    if mean_mae < best_error:
                        best_error = mean_mae
                        best_lambda_l2 = lambda_l2

                    errors.append(f"Lambda L2: {lambda_l2}, MAE: {mean_mae}")
                else:
                    mean_auc = max(cv_results['valid auc-mean'])
                    if mean_auc > best_error:
                        best_error = mean_auc
                        best_lambda_l2 = lambda_l2

            # Train the model with the best lambda_l1
            params['lambda_l2'] = best_lambda_l2
            bst_lgb = lgb.train(params, lgbtrain, 200, valid_sets=[lgbval])


        else:
            cv_results = lgb.cv(params, lgbtrain, return_cvbooster=True, stratified= False, nfold= 10)


        # Get the best iteration
        #best_iteration = np.argmax(bst_lgb['valid l1-mean'])

        # Train the final model with the best iteration
        bst_lgb = lgb.train(params, lgbtrain, 200,  valid_sets= [lgbval])

        # Make predictions using the final model
        lgb_pred = bst_lgb.predict(X_val, num_iteration=bst_lgb.best_iteration)
        
        print(f'Feature Importance for {imps}')

        if regression:
            lgb_valid_mae = mean_absolute_error(Y_val, lgb_pred)
            errors.append([f"{imps} MAE: {lgb_valid_mae} "])
            
            # Calculate median of the training target variable
            median_prediction = [Y_train.median()] * len(Y_val)

            # Calculate MAE for the median prediction
            median_mae = mean_absolute_error(Y_val, median_prediction)
            print(f"Best MAE: {lgb_valid_mae}")
            print(f"Baseline MAE using Median: {median_mae}")

        else:
            best_auc = roc_auc_score(Y_val, lgb_pred)
            lgb_accuracy = accuracy_score(Y_val, [1 if x >= 0.5 else 0 for x in lgb_pred])
            print(f"AUC: {best_auc}")
            print(f"Accuracy: {lgb_accuracy}")

        if l1_reg:
            print(f"Best L1 Lambda: {best_lambda_l1}")

        if l2_reg:
            print(f"Best L2 Lambda: {best_lambda_l2}")

        if verbose:
        # Set a larger figure size
            plt.figure(figsize=(15, 10))

            # Use plot_importance with some additional arguments
            lgb.plot_importance(bst_lgb, 
                            max_num_features=50, # limit to top 50 features; adjust as needed
                            importance_type='split', 
                            height=0.5, 
                            figsize=(15,10),
                            title=f"Feature Importance for {imps}")

            plt.yticks(fontsize=10) # adjust font size for feature names
            plt.xticks(fontsize=12) # adjust font size for importance values
            plt.show()

        """
        if predictive:
            return X_val, Y_val, bst_lgb
        """    
        return bst_lgb
    
# Plots Prediction graph for fitted lightgbm REGRESSION model
def light_predict(data, model, target, regression = True):
    cols = [col for col in data.columns if col not in ['date', target]]
    X_val = data.iloc[int(len(data)*0.8):, :][cols]
    Y_val = data.iloc[int(len(data)*0.8):, :][target]

    lgb_pred = model.predict(X_val, num_iteration=model.best_iteration)

    if regression:
        # Plotting with Plotly
        fig = go.Figure()

        # Add actual values trace
        fig.add_trace(go.Scatter(x=Y_val.index, y=Y_val, mode='lines', name='Actual', line=dict(color='blue')))

        # Add predicted values trace
        fig.add_trace(go.Scatter(x=Y_val.index, y=lgb_pred, mode='lines', name='Predicted', line=dict(color='red')))

        # Set title and labels
        fig.update_layout(title='Actual vs Predicted (PX M1-M2)', xaxis_title='date', yaxis_title='Value')

    else:
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(Y_val, lgb_pred)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve using Plotly
        trace0 = go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}')
        trace1 = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash'))

        layout = go.Layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis=dict(title='False Positive Rate'),
            yaxis=dict(title='True Positive Rate'),
            showlegend=True
        )

        fig = go.Figure(data=[trace0, trace1], layout=layout)
    fig.show()



## VISUALIZING ##

# Plotly graphs
def plotly_multi_graph(data, title, zoomed = False):

    titles = data.iloc[:, 1:]

    layout = go.Layout(
    margin=go.layout.Margin(
            l=0, #left margin
            r=0, #right margin
            b=0, #bottom margin
            t=0, #top margin
        )
    )

    if zoomed:
        max_cols = 2
        max_rows = math.ceil(len(data.columns[1:]) / max_cols)

    else:
        max_cols = 3
        max_rows = math.ceil(len(data.columns[1:]) / max_cols)

    fig = make_subplots(
        rows= max_rows, cols= max_cols,
        subplot_titles=(data.columns[1:]))
    
    x = data['date']
    
    
    curr_row = 1
    curr_col = 1

    for prod in data.columns[1:]:
        
        if curr_col > max_cols:
            curr_col = 1
            curr_row += 1
        

        fig.add_trace(go.Scatter(x=x, y = data[f"{prod}"], name= f"{prod}" ),
                    row= curr_row, col= curr_col)
        curr_col+=1

    if zoomed:
        width = 1000
        height = 240 * max_rows
        fig.update_layout(height=height, width= width,
                    title_text= title)
    else:
        width = 1500
        height = 240 * max_rows
        fig.update_layout(height=700, width= width,
                        title_text= title)

    fig.show()


def plot_multi_graphs(df):
    # Determine the number of columns (excluding the first column)
    num_columns = len(df.columns) - 1

    # Create subplots based on the number of columns
    fig, axes = plt.subplots(1, num_columns, figsize=(20, 7))

    # Iterate through columns and plot each one in a separate subplot
    for i, column in enumerate(df.columns[1:]):  # Start from the second column
        ax = axes[i] if num_columns > 1 else axes  # Use a single axes if there's only one column
        df.plot(x=df.columns[0], y=column, ax=ax, label=column)

        # Set titles and labels for each subplot
        ax.set_title(f'{column}')
        ax.set_xlabel(df.columns[0])
        #ax.set_ylabel('Values')
        #ax.legend()

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_multiple_lines(df):
    # Determine the number of columns (excluding the first column)
    num_columns = len(df.columns) - 1

    # Create a shared x-axis
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Iterate through columns and plot each one
    for i, column in enumerate(df.columns[1:]):  # Start from the second column
        df.plot(x=df.columns[0], y=column, ax=ax, label=column)

    # Set titles and labels
    ax.set_title('Arbs & Prices')
    ax.set_xlabel(df.columns[0])
    #ax.set_ylabel('Values')

    # Show a legend
    ax.legend()

    # Show the plot
    plt.show()

# Plot heatmap against a single variable
def plot_single_heatmap(data):
    if isinstance(data, pd.Series):
        # Convert series to dataframe for visualization
        target_variable = data[data == 1].index[0]
        df = data.reset_index()
        df.columns = [f'{target_variable}', 'Correlation']
    elif isinstance(data, pd.DataFrame):
        target_variable = data.columns[0]
        df = data.copy()
        df.columns = [f'{target_variable}', 'Correlation']
    else:
        raise ValueError("Input must be a pandas Series or DataFrame.")
    
    # Sort values by correlation magnitude
    df = df.sort_values(by='Correlation', ascending=False)
    
    # Drop the target variable from the dataframe
    df = df[df[f'{target_variable}'] != target_variable]
    
    # Create the heatmap
    plt.figure(figsize=(8, 20))
    sns.heatmap(df.set_index(f'{target_variable}'), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5, cbar=False)
    plt.title(f"Correlation with {target_variable}")
    plt.yticks(rotation=0)
    plt.show()

# Dynamic Rolling Corr Graph
def plot_dyn_corr_graph (data , title, corr_column, hours = False, days= False, years = False):
    df = data
    if hours:
        window = str(hours) + 'HR'
    if days:
        window = str(days) + 'D'
    if years:
        window = str(years) + 'Y'
    fig = px.line(df, x=df.index, y=corr_column, 
                  title = f'{title}: {window} Rolling Correlation').update_layout(
                  xaxis_title="Date", yaxis_title="Correlation")
    #fig.update_traces(line_color='#bd2323', line_width=1.75)

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1D",
                        step="day",
                        stepmode="backward"),
                    dict(count=6,
                        label="1H",
                       step="month",
                      stepmode="backward"),
                    dict(count=1,
                        label="YTD",
                       step="year",
                         stepmode="todate"),
                 dict(count=1,
                      label="1Y",
                       step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date",
            
        )
    )
    fig.show()

# PLot simple graph
def plot_single (data , title, col):
    df = data
    fig = px.line(df, x=df.index, y=col, 
                  title = f'{title}').update_layout(
                  xaxis_title="Date", yaxis_title=col)
    #fig.update_traces(line_color='#bd2323', line_width=1.75)

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1D",
                        step="day",
                        stepmode="backward"),
                    dict(count=6,
                        label="1H",
                       step="month",
                      stepmode="backward"),
                    dict(count=1,
                        label="YTD",
                       step="year",
                         stepmode="todate"),
                 dict(count=1,
                      label="1Y",
                       step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date",
            
        )
    )
    fig.show()

# Single Adjustable Graph
def extract_date_ranges(data_list, remove_bids_after_band=None):
    shaded_positive = []
    shaded_negative = []

    # Iterate over each dictionary in the list
    for item in data_list:
        start_date = item.get('start_time')
        end_date = item.get('end_time')
        capacity_adjustment = item.get('capacity_adjustment', 0)

        # Convert start and end dates to datetime objects with the correct format
        if start_date:
            start_date = datetime.datetime.strptime(start_date, "%Y/%m/%d %H:%M:%S")
        if end_date:
            end_date = datetime.datetime.strptime(end_date, "%Y/%m/%d %H:%M:%S")

        # Determine whether to categorize as positive or negative
        if capacity_adjustment > 1:
            shaded_positive.append((start_date, end_date))
        elif capacity_adjustment < 0 or remove_bids_after_band is not None:
            shaded_negative.append((start_date, end_date))
        else:
            shaded_negative.append((start_date, end_date))

    return [shaded_positive, shaded_negative]


def plot_single_adj(df, title, y_title, cols, y2=False, y2_col=None, shaded=None, scatter=False, width=800, height=600):
    if not scatter:
        fig = make_subplots(specs=[[{"secondary_y": y2}]])

        # Add secondary y-axis data
        if y2:
            if isinstance(y2_col, list):
                for var in y2_col:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[var],
                        mode='lines',
                        name=var,
                        yaxis='y2'
                    ), secondary_y=True)
            else:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[y2_col],
                    mode='lines',
                    name=y2_col,
                    yaxis='y2'
                ), secondary_y=True)

        # Create traces for each line
        for col in cols:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col
            ))

        # Debugging: Print shaded data
        #print("Shaded Data:", shaded)

        # Add shaded regions for positive and negative values
        if shaded and isinstance(shaded, list):
            if len(shaded) == 2:
                # Negative values
                for start_date, end_date in shaded[1]:
                    #print(f"Adding negative shaded region from {start_date} to {end_date}")
                    fig.add_vrect(
                        x0=start_date, x1=end_date,
                        fillcolor="LightSalmon", opacity=0.5,
                        layer="below", line_width=0,
                    )

                # Positive values
                for start_date, end_date in shaded[0]:
                    #print(f"Adding positive shaded region from {start_date} to {end_date}")
                    fig.add_vrect(
                        x0=start_date, x1=end_date,
                        fillcolor="LightGreen", opacity=0.5,
                        layer="below", line_width=0,
                    )
            else:
                # Default to positive if the list is a single group
                for start_date, end_date in shaded:
                    fig.add_vrect(
                        x0=start_date, x1=end_date,
                        fillcolor="LightGreen", opacity=0.5,
                        layer="below", line_width=0,
                    )

        # Update layout for more granular timestamps and custom size
        fig.update_layout(
            hoverlabel_bgcolor='#DAEEED',
            title=title,
            title_font_size=15,
            title_font_color="darkblue",
            title_x=0.46,
            xaxis=dict(
                tickfont_size=10,
                tickangle=270,
                showgrid=True,
                zeroline=True,
                showline=True,
                tickformat="%b %d\n%H:%M",  # Display month, day, hour, and minute
                dtick="H1"  # Set to show hourly ticks; adjust as needed
            ),
            legend=dict(orientation='h', xanchor="center", x=0.45, y=1.05),
            yaxis=dict(title=y_title),
            width=width,  # Set custom width
            height=height,  # Set custom height
            margin=go.layout.Margin(l=2, r=40, b=10, t=70)
        )
    else:
        # Create a scatter plot
        fig = go.Figure(data=go.Scatter(x=df.index, y=df[cols], mode='markers'))

        # Update layout with custom size
        fig.update_layout(
            title=title,
            xaxis_title='date',
            yaxis_title=y_title,
            width=width,  # Set custom width
            height=height,  # Set custom height
            xaxis=dict(
                tickformat="%b %d\n%H:%M",  # Display month, day, hour, and minute
                dtick="H1"  # Set to show hourly ticks; adjust as needed
            )
        )

    fig.show()



# Create Box & Whisker plot
def box_whisk(df, y_col, x_col, ptitle, sorted = False):
    # Create the box plot
    fig = go.Figure()

    sorted_categories = df[x_col].unique()

    if sorted:
        # Sort the unique categories in ascending order
        sorted_categories = sorted(df[x_col].unique(), key=lambda x: (x is None, x))

    # Add a box trace for each sorted category of the x_col
    for category in sorted_categories: #[:10]:
        category_data = df[df[x_col] == category][y_col]
        if len(category_data) < 3:
            continue
        fig.add_trace(go.Box(y=category_data, name=str(category)))

    # Update the layout
    fig.update_layout(
        title= f'{ptitle}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        xaxis=dict(
            categoryorder='array',  # This specifies that the x-axis categories should appear in the order given
            categoryarray=sorted_categories  # This is the order for the categories to appear in
        )
    )

    # Show the plot
    fig.show()


# Data Pulling
    
# Bloomberg Data Pulling
def request_data(tickers, fld, pull_type, date_range = None):
    """
    pull_type: bdh (Historical Data)
               bdp (Current Data)
    fld: str or list of str
    tickers: tickers in bloomberg ticker format
    fld: bdh: 'PX_LAST'
         bdp: 'PX_Bid', 'PX_Ask'    

    EXAMPLE USAGE
    -----
    start_date = datetime.now() + timedelta(days = -10*5)
    # Convert to YYYYMMDD format
    start_date = str(start_date.year)+f"{start_date.month:02}"+str(f"{start_date.day:02}")
    end_date = str(datetime.now().year)+f"{datetime.now().month:02}"+str(f"{datetime.now().day:02}")
    tickers = ['ECCPEMUY Index', 'CPI YOY Index','SHSPSCFI Index', 'WCIDCOMP Index', 'SHSPCCFI Index', 'XSICFENE Index', 'XSICFEUW Index']
    fld = ['PX_LAST']
    shipping = request_data(tickers, fld, start_date, end_date)    
    -----

    """

    if pull_type == 'bdh':
        for i in tickers:
    #         print(i)
            resp = requests.post(f'https://172.19.16.230:443/{pull_type}/',
                                data={'tickers':i,
                                    'flds': fld,
                                        'start_date': min(date_range),
                                        'end_date': max(date_range)
                                    },
                                verify = False)
            print(resp)

            # Pulling historical data
            if i == tickers[0]:
                data_first = pd.DataFrame(resp.json()).T
                #data_first['date'] = pd.to_datetime(data_first.reset_index()['date'])
                data_first = data_first.set_index('date') #.rename({fld: i}, axis=1)
                data_first = data_first.rename(columns={'PX_LAST': f'{i}'})
                #data_.index = pd.to_datetime(data_.index)
                data_first = data_first.drop('ticker', axis = 1)
                #dfs.append(data_first)

            else:
                data_ = pd.DataFrame(resp.json()).T
                #data_['date'] = pd.to_datetime(data_.reset_index()['date'])
                data_ = data_.set_index('date') #.rename({fld: i}, axis=1)
                data_ = data_.rename(columns={'PX_LAST': f'{i}'})
                #data_.index = pd.to_datetime(data_.index)
                data_ = data_.drop('ticker', axis = 1)
                data_first = pd.merge(data_first, data_, left_index=True, right_index=True, how='left')
                #dfs.append(data_)
            
            
    # Pulling current data
    if pull_type == 'bdp':
        #ate_range = pd.date_range(datetime.today(), periods=1, freq='S')
        frame_store = []

        for tick in tickers:
            ba_data = pd.DataFrame()
            data_first = pd.DataFrame(columns= fld, index=pd.Index(date_range, name='Date') )
            print(tick)
            for field in fld : 
                #print(field)
                resp = requests.post(f'https://172.19.16.230:443/{pull_type}/',
                                        data= {'tickers': f'{tick}',
                                                'flds': f'{field}',
                                                },
                                        verify = False)
                
                
                if fld.index(field)> len(fld)-2:
                    ba_data = pd.concat([ba_data, pd.DataFrame(resp.json()).T.set_index('field')])
                    data_first[f'{field}'] = ba_data.loc[f'{field}', 'value']
                    break
                try:
                    #print(pd.DataFrame(resp.json()).T)
                    ba_data = pd.DataFrame(resp.json()).T.set_index('field')
                    data_first[f'{field}'] = ba_data.loc[f'{field}', 'value']
                except:
                    print('Cant get for this contract')
                    pass    
            data_first.index = [ba_data.loc[f'{field}', 'ticker']]

            #print(ba_data.loc[f'{field}', 'ticker'])
            data_first['PX_Mid'] = (data_first['PX_Ask'] + data_first['PX_Bid'])/2
            
            frame_store.append(data_first)
        
        data_first = pd.concat(frame_store)
        

    return data_first