from __main__ import *

#######################################################
# Define functions
#######################################################
def remove_missing_targets(this_data,target_var):
    '''
    arg: target
    outputs: raw data without rows where target is missing
    '''
    this_data = this_data[~this_data[target_var].isnull()]
    this_data = this_data[~this_data["Date"].isnull()]
        
    return this_data
    

def treat_missing_feature_values_adjusted(my_verbose, this_data):
    '''
    outputs: interpolates missing values
    '''
    cols_to_adj = ["Swiss_Francs_Index", "EURO_Index", "Yen_Index"]
    this_data[cols_to_adj] = this_data[cols_to_adj].replace({0:np.nan})
    this_data.loc[~(this_data['Crude_Oil_Futures'] > 0), 'Crude_Oil_Futures']=np.nan    
    
    this_data = this_data.interpolate(method='spline', order=1)
    
    if my_verbose==True:
        print("\nMissing values have been treated")
    
    return this_data


def detrend_data(this_data,my_verbose):
    '''
    Fits polynomials to trend lines and calculates difference
    '''
    dict_degrees = {
        'SPDR_Gold_Shares': 2,
        'Gold_Futures': 2,
        'Crude_Oil_Futures': 4,
        'Palladium_Futures': 3,
        'Platinum_Futures': 4,
        'Copper_Futures': 4,
        'SP_500': 4,
        'Russell_2000': 4,
        'US_Dollar_Index': 4,
        'Swiss_Francs_Index': 3,
        'EURO_Index': 3,
        'Yen_Index': 4
    }

    X_vals = this_data.index
    X_vals = np.reshape(X_vals, (len(X_vals), 1))
    detrended_data = pd.DataFrame()
    detrended_data["Date"] = this_data.Date.values
    this_data = this_data[[x for x in this_data.columns if x!="Date"]] # and x!="SPDR_Gold_Shares"]]

    for this_col in this_data.columns.tolist():
        detrended_vals = []
        if my_verbose!=False:
            print("\n", this_col)
        raw_feature = this_data[this_col].values.reshape(-1,1)
        pf = PolynomialFeatures(degree=dict_degrees[this_col])
        Xp = pf.fit_transform(X_vals)
        md2 = LinearRegression()
        md2.fit(Xp, raw_feature)
        trendp = md2.predict(Xp)
        
        if my_verbose!=False:
            plt.plot(X_vals, raw_feature)
            plt.plot(X_vals, trendp)
            plt.legend(['data', 'polynomial trend'])
            plt.show()

        detrended_data["{}_polynomial_transform".format(this_col)] = trendp

        detrpoly = [raw_feature[i] - trendp[i] for i in range(0, len(raw_feature))]
        
        if my_verbose!=False:
            plt.plot(X_vals, detrpoly)
            plt.title('polynomially detrended data')
            plt.show()

        r2 = r2_score(raw_feature, trendp)
        rmse = np.sqrt(mean_squared_error(raw_feature, trendp))
        
        if my_verbose!=False:
            print('r2:', r2)
            print('rmse', rmse)

        for i in detrpoly:
            detrended_vals.append(i[0])
        detrended_data["{}_detrended".format(this_col)] = detrended_vals
        
    return detrended_data


def remove_outliers(this_data,stdev_multiplier,my_verbose):
    '''
    Removes outliers which are beyond certain threshold based on st dev
    '''
    detrended_features = [x for x in this_data.columns.tolist() if x != "Date" and x.endswith("detrended")]

    for this_feature in detrended_features:
        this_features_mean, this_features_stdev = np.mean(this_data[this_feature].values), np.std(this_data[this_feature].values)
        this_features_lowerbound, this_features_upperbound = this_features_mean - (stdev_multiplier*this_features_stdev), this_features_mean + (stdev_multiplier*this_features_stdev)
        this_data['{}_is_outlier'.format(this_feature)] = this_data.apply(lambda row: 'Is_Outlier' if row[this_feature]<this_features_lowerbound or  row[this_feature]>this_features_upperbound else 'Not_Outlier',axis=1)

    this_data.reset_index(inplace=True)
    
    if my_verbose!=False:
        for this_feature in detrended_features:
            plt.figure()#figsize=(12,5))
            sns.scatterplot(data=this_data, x="index", y=this_feature, hue="{}_is_outlier".format(this_feature))
        
    this_data = this_data.replace("Is_Outlier",1)
    this_data = this_data.replace("Not_Outlier",0)
    outlier_cols = [x for x in this_data.columns.tolist() if x.endswith("outlier")]
    this_data["Outlier_Indicator"] = this_data[outlier_cols].sum(axis = 1, skipna = True)
    this_data = this_data[this_data["Outlier_Indicator"]==0]
    cols_keep = [x for x in this_data.columns.tolist() if x not in outlier_cols and x!= "index" and x!="Outlier_Indicator"]
    this_data = this_data[cols_keep]
    
    return this_data


def calculate_macd_and_spreadvssignal(my_verbose, this_data, cols_to_calculate):
    '''
    outputs: dataframe with macd and macd vs spread for each shifted_col
    '''    
    
    for this_col in cols_to_calculate:

        exp1 = this_data[this_col].ewm(span=12, adjust=False).mean()
        exp2 = this_data[this_col].ewm(span=26, adjust=False).mean()
        macd = exp1-exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_signal_spread = macd - signal

        this_data["{}_macd".format(this_col)] =  macd.values #.tolist()
        this_data["{}_macd_signal_spread".format(this_col)] =  macd_signal_spread.values #.tolist()
    
    if my_verbose==True:
        print("\nMACD and spread computed")
        
    #### transformed_data.to_csv(r'MACD.csv')
    
    return this_data


def calculate_moving_averages(my_verbose, this_data, cols_to_calculate):
    '''
    outputs: dataframe with spread of price vs SMA and EMA, with rows with missing SMA removed
    '''    

    for this_col in cols_to_calculate:
        this_data['{}/15SMA'.format(this_col)] = (this_data[this_col]/(this_data[this_col].rolling(window=15).mean()))-1
        this_data['{}/30SMA'.format(this_col)] = (this_data[this_col]/(this_data[this_col].rolling(window=30).mean()))-1
        this_data['{}/60SMA'.format(this_col)] = (this_data[this_col]/(this_data[this_col].rolling(window=60).mean()))-1
        this_data['{}/90SMA'.format(this_col)] = (this_data[this_col]/(this_data[this_col].rolling(window=90).mean()))-1
        #this_data['{}/180SMA'.format(this_col)] = (this_data[this_col]/(this_data[this_col].rolling(window=180).mean()))-1



    for this_col in cols_to_calculate:
        this_data['{}/90EMA'.format(this_col)] = (this_data[this_col]/(this_data[this_col].ewm(span=90,adjust=True,ignore_na=True).mean()))-1
        #this_data['{}/180EMA'.format(this_col)] = (this_data[this_col]/(this_data[this_col].ewm(span=180,adjust=True,ignore_na=True).mean()))-1    


    this_data = this_data.iloc[179:] # take from row 181 onwards,otherwise SMA has null values
    
    if my_verbose==True:
        print("\nSpreads vs moving averages computed")
    
    #### transformed_data.to_csv(r'SMA_EMA.csv')
    
    return this_data


def scale_data(this_data, target_var):
    '''
    scales features and targets separately
    '''        

    selected_cols = [x for x in this_data.columns.tolist() if x!="Date" and x!=target_var]

    # fit on training & validation, transform training & validation, and true
    this_training_and_validation_data = this_data.iloc[:-90]
    this_true_data = this_data.iloc[-90:]

    this_x_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_this_training_and_validation_data_x = this_x_scaler.fit_transform(this_training_and_validation_data[selected_cols])
    scaled_this_true_data_x = this_x_scaler.transform(this_true_data[selected_cols])
    scaled_this_full_data_x = np.append(scaled_this_training_and_validation_data_x, scaled_this_true_data_x, axis=0)
    scaled_data_x = pd.DataFrame(data=scaled_this_full_data_x,columns=selected_cols)
    scaled_data_x.reset_index(drop=True,inplace=True)
    
    this_y_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_this_training_and_validation_data_y = this_y_scaler.fit_transform(this_training_and_validation_data[target_var].values.reshape(-1, 1))
    scaled_this_true_data_y = this_y_scaler.transform(this_true_data[target_var].values.reshape(-1, 1))
    scaled_this_full_data_y = np.append(scaled_this_training_and_validation_data_y, scaled_this_true_data_y, axis=0)
    scaled_data_y = pd.DataFrame(data=scaled_this_full_data_y,columns=[target_var])
    scaled_data_y.reset_index(drop=True,inplace=True)
    
    scaled_data = pd.merge(scaled_data_x, scaled_data_y, left_index=True, right_index=True)

    return scaled_data, this_y_scaler


def inverse_scale_target(this_scaler,this_data,target_var):
    '''
    function to descale target
    ''' 
    descaled_data = this_scaler.inverse_transform(this_data)

    descaled_data = pd.DataFrame(data=descaled_data,columns=[target_var])
    
    return descaled_data


def shift_values(my_verbose, this_data, forecasting_horizon, target_var):
    '''
    arg:
    - forecasting horizon
    - target_var. We still shift the target var so that we can calculate lagged macd and spread
    '''    
    this_data = this_data.reset_index()
    this_data.drop("index",axis=1,inplace=True)
    df_appended = this_data.copy()
    placeholder_shifted_data = this_data.copy()
    cols_to_shift = [x for x in this_data.columns.tolist() if x!="Date"]
    
    if my_verbose==True:
        print("\nShifting horizon:", forecasting_horizon, "\n")    

    for this_col in cols_to_shift:
        if my_verbose==True:
            print("Shifting column:", this_col)
        lagged_feature = []
        temp_df = this_data[[this_col]].copy()
        temp_shifted = pd.DataFrame(temp_df[this_col].shift(+forecasting_horizon))
        placeholder_shifted_data = placeholder_shifted_data.join(temp_shifted.rename(columns=lambda x: x + "_shifted"))
        if this_col != target_var:
            placeholder_shifted_data.drop(this_col,axis=1,inplace=True)
        
    placeholder_shifted_data = placeholder_shifted_data.iloc[forecasting_horizon:]
    
    return placeholder_shifted_data
    
    
def calculate_corr(target_var,min_corr,this_data):    
    '''
    calculates correlation and returns features above a certain threshold
    ''' 
    temp_df = pd.DataFrame(this_data.corr(method="spearman")[np.abs(this_data.corr(method="spearman"))>=min_corr].iloc[0,:].copy())
    temp_df.sort_index(inplace=True)
    temp_df = temp_df[~temp_df[target_var].isnull()]
    
    if len(temp_df)<7:
        temp_df = pd.DataFrame(this_data.corr(method="spearman")[np.abs(this_data.corr(method="spearman"))>=0.5].iloc[0,:].copy())
        temp_df.sort_index(inplace=True)
        temp_df = temp_df[~temp_df[target_var].isnull()]
        print("\nReduced min_corr to 0.5 for this horizon")
        
    if len(temp_df)<7:
        temp_df = pd.DataFrame(this_data.corr(method="spearman")[np.abs(this_data.corr(method="spearman"))>=0.4].iloc[0,:].copy())
        temp_df.sort_index(inplace=True)
        temp_df = temp_df[~temp_df[target_var].isnull()]
        print("\nReduced min_corr to 0.4 for this horizon")
        
    if len(temp_df)<7:
        temp_df = pd.DataFrame(this_data.corr(method="spearman")[np.abs(this_data.corr(method="spearman"))>=0.3].iloc[0,:].copy())
        temp_df.sort_index(inplace=True)
        temp_df = temp_df[~temp_df[target_var].isnull()]
        print("\nReduced min_corr to 0.3 for this horizon")
        
    if len(temp_df)<7:
        temp_df = pd.DataFrame(this_data.corr(method="spearman")[np.abs(this_data.corr(method="spearman"))>=0.0].iloc[0,:].copy())
        temp_df.sort_index(inplace=True)
        temp_df = temp_df[~temp_df[target_var].isnull()]
        print("\nReduced min_corr to 0.0 for this horizon")
    
    return temp_df

def plot_corr_heatmap(this_data,target_var):
    if len(this_data)>1:
        sns.heatmap(this_data.sort_values(by=target_var,ascending=False))
        plt.show()
    else:
        print("No features with correlation above ", min_corr)
    
    return plt.show()


def plot_scaled_scatter(this_data, selected_col, target):

    this_scaler = MinMaxScaler(feature_range=(0, 1))
    selected_data = this_scaler.fit_transform(this_data[[target, selected_col]])
    selected_data = pd.DataFrame(selected_data,columns=[target, selected_col])

    plt.figure(figsize=(16,5)) 
    plt.title('Scatter of {} (X) with {} (Y)'.format(target, selected_col))
    plt.scatter(selected_data[target].values, selected_data[selected_col].values)
    
    return plt.show()


def split_train_validation_true_data(target_var, this_cols_retain, this_training_and_validation_data, this_true_data, test_size):
    '''
    splits train and validation set
    '''     
    this_training_and_validation_data = this_training_and_validation_data[this_cols_retain]
    this_true_data = this_true_data[this_cols_retain]

    this_cols_retain.remove(target_var)
    Xs = this_training_and_validation_data[this_cols_retain]
    Ys = this_training_and_validation_data[target_var]

    X_train, X_val , y_train, y_val = train_test_split(Xs, Ys, test_size=test_size, random_state=42, shuffle=True)
    
    return X_train, X_val , y_train, y_val, this_training_and_validation_data, this_true_data


def mean_absolute_percentage_error(y_true, y_pred): 
    '''
    function to calculate MAPE
    '''    
    temp_mape = []
    for this_true, this_pred in zip(y_true, y_pred):
        temp_val = np.abs((this_true - this_pred) / this_true)
        temp_mape.append(temp_val)

    return np.mean(temp_mape) * 100, temp_mape



def train_and_select_model(my_verbose, metric, this_training_x, this_validation_x, this_training_y, this_validation_y, target_var, this_y_scaler):
    '''
    Train model, evaluate on validation set
    '''

    # Compile models
    # tune ET, RF: https://stackoverflow.com/a/22546016/6877740
    models = []
#     models.append(('LR', LinearRegression()))
#     models.append(('LASSO', Lasso()))
#     models.append(('EN', ElasticNet()))
#     models.append(('KNN', KNeighborsRegressor()))
#     models.append(('CART', DecisionTreeRegressor()))
#     models.append(('SVR', SVR()))
#     models.append(('AB', AdaBoostRegressor()))
    models.append(('GBM', GradientBoostingRegressor(n_estimators=50,max_depth=5,min_samples_leaf=2)))
    models.append(('RF', RandomForestRegressor(n_estimators=50,max_depth=5,min_samples_leaf=2)))
    models.append(('ET', ExtraTreesRegressor(n_estimators=50,max_depth=5,min_samples_leaf=2)))
    model_names = [x[0] for x in models]

    list_rms = []
    list_mapes = []
    list_rsq = []
    list_predictions = []

    descaled_validation_actual_target = inverse_scale_target(this_y_scaler,this_validation_y.values.reshape(-1, 1),target_var)
    descaled_validation_actual_target = descaled_validation_actual_target.values.reshape(-1,1)
    y_true = descaled_validation_actual_target    


    for this_model in models:
        this_model_name = this_model[0]
        this_regressor = this_model[1]

        reg = this_regressor.fit(this_training_x.values, this_training_y.values.reshape(-1,1))

        # evaluate model on validation
        predictions = reg.predict(this_validation_x.values)
        predictions = predictions.reshape(-1,1)
        descaled_validation_predicted_target = inverse_scale_target(this_y_scaler,predictions,target_var)
        descaled_validation_predicted_target = descaled_validation_predicted_target.values.reshape(-1,1)        

        # compute errors        
        y_pred = descaled_validation_predicted_target
        list_predictions.append(y_pred)
        rms = sqrt(mean_squared_error(y_true, y_pred))
        mape, apes = mean_absolute_percentage_error(y_true, y_pred)
        rsq = r2_score(y_true, y_pred)

        list_rms.append(rms)
        list_mapes.append(mape)
        list_rsq.append(rsq)

    if my_verbose==True:
        print("\nModels trained complete")

    if metric == "RMSE":
        errors_list = list_rms
        val, idx  = min((val, idx) for (idx, val) in enumerate(list_rms))

        print("\nLowest validation {} of: {:.2f}".format(metric, val))

    elif metric == "MAPE":
        errors_list = list_mapes
        val, idx  = min((val, idx) for (idx, val) in enumerate(list_mapes))

        print("\nLowest validation {} of: {:.2f}%".format(metric, val))

    elif metric == "RSQ":
        errors_list = list_rsq
        val, idx  = max((val, idx) for (idx, val) in enumerate(list_rsq))

        print("\nHighest validation {} of: {:.2f}%".format(metric, val))        
        
        
    best_y_pred = list_predictions[idx]
    best_model = models[idx]
    best_error = val
    best_rsq = list_rsq[idx]
    
    # temp_df = pd.DataFrame(best_y_pred,columns=["y_pred"])
    # temp_df["y_true"] = y_true
    # temp_df.to_csv("checks_v2.csv")

    return y_true, best_y_pred, best_model, best_error, best_rsq


def plot_scatter_actuals_predicted(this_actuals, this_predicted):
    '''
    plot actuals vs predicted. The closer to the diagonal, the better.
    ''' 
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(this_actuals, this_predicted, c='black')
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    
    return plt.show()


def predict_test(this_model, this_true_data, this_y_scaler, target_var, environment):
    '''
    predict based on test set. Always uses last time stamp in training set.
    '''     
    this_model_name = this_model[0]
    this_regressor = this_model[1]
    
    x_cols = [x for x in this_true_data.columns.tolist() if x != target_var]
    X_test = this_true_data[x_cols]
    
    if environment == "PRD":
        y_test_actual = None
        y_test_actual_descaled = None

    elif environment == "QAS":
        y_test_actual = this_true_data[target_var].values.reshape(-1,1)[0]   

        # descale target
        descaled_test_actual_target = inverse_scale_target(this_y_scaler,y_test_actual.reshape(-1, 1),target_var)
        descaled_test_actual_target = descaled_test_actual_target.values.reshape(-1,1)
        y_test_actual_descaled = descaled_test_actual_target[0]      

    # get prediction
    reg = this_regressor
    predictions = reg.predict(X_test.values)
    predictions = predictions.reshape(-1,1)[0]
    descaled_test_predicted_target = inverse_scale_target(this_y_scaler,predictions.reshape(-1, 1),target_var)
    descaled_test_predicted_target = descaled_test_predicted_target.values.reshape(-1,1)        
    y_pred = descaled_test_predicted_target[0]
    
    return y_test_actual, y_test_actual_descaled, predictions, y_pred, this_model_name


def generate_test_results(this_test_results,this_prediction_date):
    '''
    generates a dataframe with test results
    '''     
    this_actual = this_test_results["Actuals - Descaled"].values
    this_pred = this_test_results["Predicted - Descaled"].values
    
    this_test_results["APE"] =  np.abs(this_actual - this_pred) / this_actual * 100
    test_MAPE = this_test_results.APE.mean()
    test_rsq = r2_score(this_actual, this_pred)
    test_rms = sqrt(mean_squared_error(this_actual , this_pred))
    
    prediction_date = [this_prediction_date]*len(this_test_results)
    
    #print("MAPE: {:.2f}%, RSQ: {:.2f}%, RMSE: {:.2f}".format(test_MAPE[0], test_rsq, test_rms))
    print("MAPE: {:.2f}%".format(test_MAPE[0]))
    
    return this_test_results


def plot_test_results(this_test_results):
    '''
    Plots test results
    '''     
    plt.figure(figsize=(16,5)) 
    plt.title('SPDR Gold Shares (USD): Actuals vs Predicted')
    plt.plot(list(range(0,len(this_test_results))), this_test_results["Actuals - Descaled"].values, label = "Actual")
    plt.plot(list(range(0,len(this_test_results))), this_test_results["Predicted - Descaled"].values, label = "Predicted")
    plt.legend()
    plt.show()    

    plt.figure(figsize=(16,5)) 
    plt.title('SPDR Gold Shares (USD): % Error (Actual vs Predicted)')
    plt.plot(list(range(0,len(this_test_results))), this_test_results["APE"].values, label = "% Error")
    plt.legend()
    plt.show()

    
def inspect_issues(this_test_results, this_cached_scaled_data):
    '''
    qas function to pick out samples with highest MAPE
    '''     
    this_horizon = test_results.sort_values(by="APE",ascending=False).iloc[0].Horizon
    this_APE = test_results.sort_values(by="APE",ascending=False).iloc[0].APE[0]
    print("Highest APE at horizon {}, with APE of {:.2f}".format(this_horizon, this_APE))
    temp_df = this_cached_scaled_data.copy()
    temp_df.reset_index(inplace=True)
    check_cols = [x[:-8] for x in dict_features[this_horizon]] # remove "shifted suffix"
    print(temp_df.iloc[-91][check_cols])

    temp_df.plot(x = 'index', y=target_var, kind = 'scatter')
    plt.axvline(x=temp_df.index.max()-(this_horizon+1))
    plt.show()

    for this_col in check_cols:
        temp_df.plot(x = 'index', y=this_col, kind = 'scatter')
        plt.axvline(x=temp_df.index.max()-(this_horizon+1))
        plt.show()