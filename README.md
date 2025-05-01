*Content*
The data is presented in a couple of formats to suit different individual's needs or computational limitations. I have included files containing 5 years of stock data (in the all_stocks_5yr.csv and corresponding folder).

The folder individual_stocks_5yr contains files of data for individual stocks, labelled by their stock ticker name. The all_stocks_5yr.csv contains the same data, presented in a merged .csv file. Depending on the intended use (graphing, modelling etc.) the user may prefer one of these given formats.

All the files have the following columns:
Date - in format: yy-mm-dd

Open - price of the stock at market open (this is NYSE data so all in USD)

High - Highest price reached in the day

Low Close - Lowest price reached in the day

Volume - Number of shares traded

Name - the stock's ticker name


----------------------------------------------------------------------------------------------------------

# Time Series Forecasting with LSTM

This project demonstrates time series forecasting using a stacked Long Short-Term Memory (LSTM) neural network, applied to historical stock data from the S&P 500 index.

## 📌 Overview

The notebook performs the following:

- Downloads the historical S&P 500 dataset from Kaggle.
- Visualizes and preprocesses the "close" stock prices.
- Scales the data using MinMaxScaler.
- Prepares data sequences for time series forecasting.
- Builds and trains a stacked LSTM model.
- Evaluates model performance using visual plots.

## 📂 Dataset

The dataset used is sourced from [Kaggle: S&P 500 Stock Data](https://www.kaggle.com/datasets/camnugent/sandp500), which includes 5 years of historical stock prices for multiple companies.

## 🛠️ Technologies

- Python
- NumPy & Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

## 🧠 Model Architecture

- Input Shape: `[samples, timesteps, features]`
- Stacked LSTM layers
- Output: Forecasted stock price for the next time step

## 📊 Evaluation

The model forecasts stock prices and visualizes the predicted vs. actual prices for analysis.

## 📝 Usage

1. Clone the repo or open the `TimeSeries_Forecasting.ipynb` notebook.
2. Make sure required libraries (`tensorflow`, `numpy`, `pandas`, etc.) are installed.
3. Run the notebook step-by-step to train and test the LSTM model.

## 📌 Note

- The notebook uses Kaggle Hub to download the dataset. Make sure you have Kaggle API credentials configured if needed.
- Adjust `time_step`, model parameters, and layers to fine-tune performance.

## 📈 Output Example

The notebook includes plots such as:

- Training loss vs epochs
- Predicted vs actual stock prices

## 📬 Contact

For questions or feedback, feel free to reach out to the author- SOHOM MUKHERJEE
www.sohommukherjee.wordpress.com

---

