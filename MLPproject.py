# Index Optimizer Code Test

##### 1. Package and Assumption #####

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from sklearn.decomposition import PCA

''' Assumptions
1. Assume 0.15% transaction costs, and no slippage for simplicity.

2. The covariance matrix is based on the last 100 trading days, the number 100 is obtained by cross-valition, assuming that when the larger percentage the top 3 PCA can explain the variance, the better the covariance matrix.

3. Due to the data availability, assuming trading at the adjust close price, the code could be easily changed to 'adjust vwap' if there is the data.
'''

##### 2. Get Data #####

# Load the CSV file '000300cons.csv' containing CSI 300 index constituent data
CSI_300 = pd.read_csv('000300cons.csv')

# Extract the values from the 'Unnamed: 4' column of the DataFrame
CSI_300 = CSI_300['Unnamed: 4'].values

# Format the extracted values as strings with leading zeros to have 6 digits
CSI_300 = ['%06d' % i for i in CSI_300]

# List of CSI 300 constituents' tickers
csi300_tickers = [stock + '.SS' if stock.startswith('6') else stock + '.SZ' for stock in CSI_300]

# Specify date range (2021-2022)
start_date = '2021-01-01'
end_date = '2022-12-31'

# Retrieve historical data using yfinance
data = yf.download(csi300_tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns_df = data.pct_change().fillna(0)

# Save returns data to a CSV file
returns_df.to_csv('csi300_returns_2022.csv')

##### 3. Risk Calculation and Signal #####

# Load historical data for CSI 300 index constituents' returns
returns_df = pd.read_csv('csi300_returns_2022.csv', index_col=0)

# Specify date range
start_date = '2022-01-01'
end_date = '2022-12-31'

# Generate business day range within the specified date range
business_days = [date for date in returns_df.index.values if date >= start_date]

# Calculate the daily covariance matrix using a rolling window of 100 business days
rolling_covariance_matrices = {}
window_size = 100

for day in business_days:
    window_loc = np.where(returns_df.index.values == day)[0][0]
    window_returns = returns_df.iloc[(window_loc + 1 - window_size):(window_loc + 1)]
    rolling_covariance_matrix = window_returns.cov()
    rolling_covariance_matrices[day] = rolling_covariance_matrix

# Sample list of dates
dates = returns_df.index.values

# Sample list of stock symbols (replace with your actual stock symbols)
stock_symbols = csi300_tickers

# Generate random alpha signals
num_dates = len(dates)
num_stocks = len(stock_symbols)

# Set a random seed for reproducibility (remove this line if you want non-reproducible results)
np.random.seed(42)

# Generate random alpha signals between -1 and 1
random_alpha_signals = np.random.uniform(low=-1, high=1, size=(num_dates, num_stocks))

# Create a DataFrame to store the random alpha signals
alpha_df = pd.DataFrame(random_alpha_signals, index=dates, columns=stock_symbols)

# Save the alpha signals DataFrame to a CSV file
alpha_df.to_csv('random_alpha_signal.csv')

# Calculate equal weights
equal_weights = np.ones(num_stocks) / num_stocks

# Create a DataFrame to store the equal weights for each date
index_weights_df = pd.DataFrame([equal_weights] * len(dates), index=dates, columns=stock_symbols)

# Save the index weights DataFrame to a CSV file
index_weights_df.to_csv('index_weights.csv')

##### 4. Markowitz Optimization and Back-test #####

# Load historical data for CSI 300 index constituents' returns, alpha signal and index weight
returns_df = pd.read_csv('csi300_returns_2022.csv', index_col=0)
alpha_signal = pd.read_csv('random_alpha_signal.csv', index_col=0)
index_weights = pd.read_csv('index_weights.csv', index_col=0)

# Calculate covariance matrix
covariance_matrix = rolling_covariance_matrices

# Number of assets
num_assets = len(returns_df.columns)

# Initial investment amount
initial_investment = 72800000  # $10 million

# Transaction fee cost (0.15%)
transaction_fee = 0.0015

# Specify maximum deviation from index weight (3%)
weight_deviation_limit = 0.03

# Turnover rate limit (15%)
turnover_limit = 0.15

# Lambda value for the return & risk trade-off
lambda_value = 0.05

# Calculate the number of trading days
num_trading_days = len(returns_df)

# Initialize lists to store portfolio values and weights
portfolio_values = []
portfolio_weights = []

# Define initial weights (equal weights)
initial_weights = np.array([1.0 / num_assets] * num_assets)

# Define first flag
first_flag = True

# Backtest loop
for day in business_days:
    # Get the index location of the day in the dataframe
    window_loc = np.where(returns_df.index.values == day)[0][0]


    # Define Markowitz objective function
    def objective(weights):
        portfolio_expected_return = np.dot(alpha_signal.iloc[window_loc].values, weights)
        portfolio_risk = np.sqrt(np.dot(np.dot(weights, covariance_matrix[day]), weights))
        return -portfolio_expected_return + portfolio_risk  # Negative for maximization


    # Define weight constraints
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    # Define weight deviation constraint
    index_weight = index_weights.iloc[window_loc].values
    weight_deviation_constraint = {'type': 'ineq', 'fun': lambda weights: weight_deviation_limit - np.sum(
        np.abs(weights - index_weight))}
    constraints.append(weight_deviation_constraint)

    # Define turnover constraint
    if not first_flag:
        turnover_rate = lambda weights: np.linalg.norm(weights - portfolio_weights[-1], 1) / 2
        turnover_constraint = {'type': 'ineq',
                               'fun': lambda weights: turnover_limit * num_assets - turnover_rate(weights)}
        constraints.append(turnover_constraint)

    # Non-negativity constraint on weights
    bounds = tuple((0, None) for _ in range(num_assets))

    # Initial guess for asset weights
    if first_flag:
        initial_weights = np.ones(num_assets) / num_assets
    else:
        initial_weights = portfolio_weights[-1]

    # Solve the optimization problem using non-convex optimization
    result = minimize(objective, initial_weights, constraints=constraints, bounds=bounds, method='SLSQP',
                      options={'disp': False})
    optimal_weights = result.x

    # Update initial weights for the next day
    initial_weights = optimal_weights

    # Calculate portfolio value for the day
    if first_flag:
        portfolio_value = initial_investment
        first_flag = False
    else:
        portfolio_value = np.dot(portfolio_values[-1], (1 + returns_df.iloc[window_loc]))

    # Calculate asset values for the day
    asset_values = portfolio_value * optimal_weights * (1 + returns_df.iloc[window_loc])

    # Apply transaction fee cost
    asset_values_after_fee = asset_values * (1 - transaction_fee)

    # Append portfolio value and weights to lists
    portfolio_values.append(asset_values_after_fee.sum())
    portfolio_weights.append(optimal_weights)

# Create a DataFrame to store backtest results
backtest_results = pd.DataFrame({'Date': business_days, 'Portfolio Value': portfolio_values})

# Save backtest results to a CSV file
backtest_results.to_csv('backtest_results_with_fee.csv', index=False)