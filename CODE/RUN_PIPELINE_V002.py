"""
Main script to run the entire pipeline.

This script runs the following:
1. Gets data from Yahoo!Finance
2. Processes and transforms data
3. Trains models and selects best model
4. Generates and saves forecasts for each horizon
5. Converts forecasts to json for consumption in flask

Further notes can be found here: https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/wiki/2.-Pipeline-explanation

This scripts expects a list containing the tickers to extract from Yahoo!Finance.
"""

#######################################################
# Load Python packages
#######################################################
from yahoofinancials import YahooFinancials
import os
import csv
import json

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from math import sqrt
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split

# https://stackoverflow.com/questions/40516661/adding-line-to-scatter-plot-using-pythons-matplotlib
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


#######################################################
# Load child scripts
#######################################################
import GET_YAHOO_DATA_V001
import RUN_MODEL_V002
import CONVERT_FORECASTS_TO_JSON_V001


#######################################################
# Cache start time
#######################################################
starting_time = datetime.datetime.now().replace(microsecond=0)


#######################################################
# Set script parameters
#######################################################
# Configuration
target_var = "SPDR_Gold_Shares"
my_verbose = False
min_horizon = 1
max_horizon = 90
min_corr = 0.65
metric = "MAPE" # MAPE or RMSE
environment = "PRD" # "QAS" or "PRD"
stdev_multiplier = 2.5

#create folders
if not os.path.exists('.\RESULTS'):
    os.makedirs('.\RESULTS')

if not os.path.exists('.\DATA'):
    os.makedirs('..\DATA')    
 

#######################################################
# Read ticker list
#######################################################

ticker_list = pd.read_excel(r'.\DATA\List of Tickers.xlsx',sheet_name = 'Main')

#######################################################
# Pull data via YAHOO FINANCE API
#######################################################

start_date = '2010-11-01'
end_date = (datetime.datetime.now() - datetime.timedelta(1)).strftime("%Y-%m-%d")

placeholder_df = GET_YAHOO_DATA_V001.create_placeholder_df(start_date, end_date)
GET_YAHOO_DATA_V001.compile_data(start_date, end_date, placeholder_df, ticker_list)


#######################################################
# Run modelling
#######################################################
  
dict_features = {}    

# load data
raw_data = pd.read_csv(r'.\DATA\RAW_DATA.csv')
#raw_data = pd.read_csv(r'.\DATA\RAW_DATA_TEST_PRD_SCRIPT.csv')
try:
    raw_data.Date = pd.to_datetime(raw_data.Date, format='%Y-%m-%d')
except ValueError:
    raw_data.Date = pd.to_datetime(raw_data.Date, format="%d/%m/%Y")
    
raw_data = RUN_MODEL_V002.remove_missing_targets(raw_data,target_var)
filled_data = RUN_MODEL_V002.treat_missing_feature_values_adjusted(my_verbose, raw_data)
after_data = filled_data.iloc[2000:] # time stamp to use
after_data.reset_index(drop=True,inplace=True)

if environment == "PRD":
    # if prd, create 90 blank rows
    for i in range(min_horizon, max_horizon+1):
        temp_list = [None,999999]
        temp_list_nils = [None] * (len(after_data.columns.tolist())-2) #add blank last row to serve as prediction placeholder
        temp_list.extend(temp_list_nils)
        after_data.loc[len(after_data)] = temp_list #using loc with index value that does not exist. Be sure to reset index first

training_and_validation_data = after_data.iloc[:-90]
true_data = after_data.iloc[-90:]

# clean outliers from training & validation, using only training & validation
detrended_data = RUN_MODEL_V002.detrend_data(training_and_validation_data,my_verbose)
detrended_data = RUN_MODEL_V002.remove_outliers(detrended_data,stdev_multiplier,my_verbose)
keep_dates = detrended_data.Date.tolist()
training_and_validation_data = training_and_validation_data[training_and_validation_data.Date.isin(keep_dates)]

# append training & validation, with true_data
# (we need to ensure all are in one so that we can create the same columns for subsequent preprocessing steps)
cleansed_data = training_and_validation_data.append(true_data)
cleansed_data.reset_index(drop=True,inplace=True)
if my_verbose == "True":
    print("Total rows: ", cleansed_data.shape[0])

# engineer new features expect for target. target is highly correlated with Gold Futures
cols_to_calculate = [x for x in cleansed_data.columns.tolist() if not x.endswith("SPDR_Gold_Shares")]
cols_to_calculate.remove("Date")
transformed_data = RUN_MODEL_V002.calculate_macd_and_spreadvssignal(my_verbose, cleansed_data, cols_to_calculate)
transformed_data = RUN_MODEL_V002.calculate_moving_averages(my_verbose, transformed_data, cols_to_calculate)

# cache features before transformation
cached_transformed_data = transformed_data.copy()
cached_descaled_data = cached_transformed_data
prediction_date = cached_transformed_data.Date.max().strftime('%Y-%m-%d')

# scale data
transformed_data, this_y_scaler = RUN_MODEL_V002.scale_data(transformed_data, target_var)

# cache the scaled data
cached_scaled_data = transformed_data

# cache y_values for true data
test_results = pd.DataFrame(columns=["Horizon","Actuals - Scaled", "Actuals - Descaled", "Predicted - Scaled", "Predicted - Descaled","Model Name"])

for forecasting_horizon in range(min_horizon, max_horizon+1):
    
    print("\n\nHORIZON:", forecasting_horizon)
    
    # shift data
    shifted_data = transformed_data.copy()
    shifted_data = RUN_MODEL_V002.shift_values(my_verbose, shifted_data, forecasting_horizon, target_var)
    
    # separate for feature selection, which is based on training & validation
    training_and_validation_data = shifted_data.iloc[:-90]
    true_data = shifted_data.iloc[-90:]
    temp_df = pd.DataFrame(columns=shifted_data.columns.tolist())
    # from true_value, select true value we are aiming to predict ie the horizon-th row in true_data
    true_data = temp_df.append(true_data.iloc[forecasting_horizon-1])
        
    # get features
    selected_features_df = RUN_MODEL_V002.calculate_corr(target_var,min_corr,training_and_validation_data)
    cols_retain = list(selected_features_df.index)
    dict_features[forecasting_horizon] = cols_retain
    print("{} features used".format(len(cols_retain)))
    
    if environment == "QAS":
        # plot_corr_heatmap uni-variate scatter
        RUN_MODEL_V002.plot_corr_heatmap(selected_features_df,target_var)
        data_for_plots = training_and_validation_data[list(selected_features_df.index)]
        selected_cols = data_for_plots.columns.tolist()
        target = "SPDR_Gold_Shares"
        selected_cols.remove(target)
        for selected_col in selected_cols:
            RUN_MODEL_V002.plot_scaled_scatter(data_for_plots, selected_col, target)    

    if my_verbose == "True":
        print("\nHorizon: ", forecasting_horizon, "\nNo. of features: ", len(cols_retain), "\n\n")
    
    # split train, validation, and true data
    X_train, X_validation , y_train, y_validation, training_and_validation_data, true_data = RUN_MODEL_V002.split_train_validation_true_data(target_var, cols_retain, training_and_validation_data, true_data, test_size=0.33)
    
    
    # train models and pick best based on metric
    y_validation_descaled, best_y_pred_descaled, best_model, best_error, best_rsq = RUN_MODEL_V002.train_and_select_model(my_verbose, metric, X_train, X_validation, y_train, y_validation, target_var, this_y_scaler)

    print("\nBest model is", best_model[0], "with {}: {:.2f}".format("RSQ", best_rsq))
    
    if environment == "QAS":
        RUN_MODEL_V002.plot_scatter_actuals_predicted(y_validation_descaled, best_y_pred_descaled)
        
    # get test results
    y_test_actual, y_test_actual_descaled, prediction_scaled, prediction_descaled, this_model_name = RUN_MODEL_V002.predict_test(best_model, true_data, this_y_scaler, target_var,environment)
    last_row = len(test_results)
    test_results.loc[last_row] = [forecasting_horizon,y_test_actual, y_test_actual_descaled, prediction_scaled, prediction_descaled, this_model_name]        
    

if (test_results["Actuals - Scaled"].sum() == cached_scaled_data.iloc[-90:].SPDR_Gold_Shares.sum()) and (environment == "QAS"):
    print("Successfully cached true values")
    
# generate results and plot graph    
if environment == "QAS":
    test_results = RUN_MODEL_V002.generate_test_results(test_results,prediction_date)
    RUN_MODEL_V002.plot_test_results(test_results)

# output results
todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
test_results.to_csv(r'.\RESULTS\saved_forecasts_{}.csv'.format(environment),index=False)

cached_descaled_data.to_csv(r'.\RESULTS\saved_descaled_data.csv',index=False)    
cached_scaled_data.to_csv(r'.\RESULTS\saved_scaled_data.csv',index=False)   
print("\nCompleted successfully")
ending_time = datetime.datetime.now().replace(microsecond=0)
CONVERT_FORECASTS_TO_JSON_V001.convert_forecasts_to_json()
print("Total elapsed time: ", ending_time-starting_time)