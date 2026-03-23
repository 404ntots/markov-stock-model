# Markov Chain Stock Price Modeling

## Overview
This project models stock price dynamics using a discrete-time Markov chain. 
Daily returns are classified into 5 states to analyze transition behavior and long-term trends.

## Data
- Source: Yahoo Finance (AAPL)
- Daily adjusted closing prices
- Converted into percentage returns

## State Definition
- S1: Significant Increase (>= +3%)
- S2: Slight Increase (+0.5% to +3%)
- S3: Stable (-0.5% to +0.5%)
- S4: Slight Decrease (-3% to -0.5%)
- S5: Significant Decrease (<= -3%)

## Methodology
- Convert prices → returns → states
- Build transition matrix
- Compute stationary distribution (power iteration)

## Results

### Transition Matrix
![Transition](images/transition_heatmap.png)

### Stationary Distribution
![Stationary](images/stationary_distribution.png)

## Key Insights
- Most probability mass is near stable states
- Extreme movements are rare
- Indicates moderate volatility

## Tech Stack
Python, NumPy, pandas, Matplotlib

## How to Run

```bash
pip install -r requirements.txt
python main.py