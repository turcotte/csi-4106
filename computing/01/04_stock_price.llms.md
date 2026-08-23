# Jupyter Notebook - Stock Prices

CSI 4106 - Fall 2025

Author

Affiliations

Marcel Turcotte [](mailto:marcel.turcotte@uottawa.ca)

School of Electrical Engineering and Computer Science

University of Ottawa

Published

January 30, 2026

# Market Capitalization of NVIDIA and Intel (2012 - 2024)

The Python library `yfinance` is often used to download stock market data.

``` python
import yfinance as yf
import matplotlib.pyplot as plt
```

Let’s define the stocks that are of interest for this analysis.

``` python
# Define the tickers for NVIDIA and Intel
tickers = ['NVDA', 'INTC']
```

Now, downloading the data to our Colab instance or local computer.

``` python
# Download the historical market data since 2012
data = yf.download(tickers, start='2012-01-01', end='2024-01-01', group_by='ticker')
```

Focusing on the closing prices.

``` python
# Extract the adjusted closing prices
nvda_data = data['NVDA']['Close']
intc_data = data['INTC']['Close']
```

Drawing.

``` python
# Plot the stock price data
plt.figure(figsize=(12, 6))
plt.plot(nvda_data.index, nvda_data, label='NVIDIA')
plt.plot(intc_data.index, intc_data, label='Intel')
plt.title('Stock Prices of NVIDIA and Intel (2012 - 2024)')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
```

Now calculating the market capitalisation.

``` python
# Fetch the number of shares outstanding (this gives the most recent value)
nvda_shares = yf.Ticker('NVDA').info['sharesOutstanding']
intc_shares = yf.Ticker('INTC').info['sharesOutstanding']

# Calculate market capitalization (Adjusted Close * shares outstanding)
nvda_market_cap = data['NVDA']['Close'] * nvda_shares
intc_market_cap = data['INTC']['Close'] * intc_shares
```

While the share prices of NVIDIA and Intel are comparable, NVIDIA’s market capitalization has experienced a significant increase since 2020, in contrast to Intel’s more stable market capitalization.

``` python
# Plot the market capitalization data
plt.figure(figsize=(12, 6))
plt.plot(nvda_market_cap.index, nvda_market_cap, label='NVIDIA')
plt.plot(intc_market_cap.index, intc_market_cap, label='Intel')
plt.title('Market Capitalization of NVIDIA and Intel (2012 - 2024)')
plt.xlabel('Date')
plt.ylabel('Market Capitalization (USD)')
plt.legend()
plt.grid(True)
plt.show()
```
