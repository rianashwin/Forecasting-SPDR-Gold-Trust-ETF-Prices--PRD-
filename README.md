# Forecasting-SPDR-Gold-Trust-ETF-Prices (PRD)
This project aims to forecast [SPDR Gold Trust ETF (NYSEARCA: GLD)](https://finance.yahoo.com/quote/GLD/profile?p=GLD) for t=+1 to t=+90. We then build a simple flask app to provide an endpoint to retrieve these forecasts.

## Background

### Preamble and data sources
Gold is typically seen as a safe-haven asset. Aside from being used for jewellery, it is also a reliable store of value. Typically, it hedges against the performance of bonds and stocks.

When building this model, we include features around alternative metals to gold (copper, palladium), as well as alternative assets such as crude oil. We also measure the strength of the stock market by including S&P 500 and the Russell 2000 index, as well as a measure of the strenght of the  USD, Japanese Yen and Swiss Francs.

Data sourced from [Yahoo! Finance](https://finance.yahoo.com/) using an API (explained in subsequent sections of this README):

| Symbol                 | Description | Reasoning | URL | 
| :---                    | --- | --- | --- |
| GLD            | SPDR_Gold_Shares | This is the target variable| https://finance.yahoo.com/quote/GLD/profile?p=GLD |
| GC=F            | Gold_Futures | Future of alternatives, indicates confidence | https://finance.yahoo.com/quote/GC=F/ |
| CL=F            | Crude_Oil_Futures | Future of alternatives, indicates confidence | https://finance.yahoo.com/quote/CL=F/ |
| PA=F            | Palladium_Futures | Future of alternatives, indicates confidence | https://finance.yahoo.com/quote/PA=F/ |
| PL=F            | Platinum_Futures | Future of alternatives, indicates confidence | https://finance.yahoo.com/quote/PL=F/ |
| HG=F            | Copper_Futures | Future of alternatives, indicates confidence | https://finance.yahoo.com/quote/HG=F/ |
| ^GSPC            | SP_500 | Performance of alternatives, for large cap| https://finance.yahoo.com/quote/%5Egspc/|
| ^RUT            | Russell_2000 | Performance of alternatives, for large cap| https://finance.yahoo.com/quote/%5ERUT/ |
| DX=F            | US_Dollar_Index | Performance of safe-haven for companies | https://finance.yahoo.com/quote/DX=F/ |
| 6S=F            | Swiss_Francs_Index | Performance of safe-haven for companies | https://finance.yahoo.com/quote/6S=F/ |
| 6E=F            | EURO_Index | Performance of safe-haven for companies | https://finance.yahoo.com/quote/6E=F/ |
| 6J=F            | Yen_Index | Performance of safe-haven for companies | https://finance.yahoo.com/quote/6J=F/ |

## Requirements and project structure

### Environment
There are a number of key packages used. Aside from the usual suspects of pandas, matplotlib, numpy, sklearn and keras, we need to use a python module that allows us to query data from Yahoo! Financials. Further notes for this package is available via [PyPI](https://pypi.org/project/yahoofinancials/). We can install this by running the following

    python -m pip install yahoofinancials
    
List of packages can be found [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/blob/main/requirements.txt). Installation can be completed using conda, as follows:
    
    conda create --name <env> --file <this file>
    
You can also create this environment from the [environment.yml](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/blob/main/environment.yml) file by running the following:

    conda env create -f environment.yml

### Project structure
The following is the expected project structure. These folders are created by the script itself, however, there is a configuration file that is required before running, indicated below.

```
├── .gitignore                          <- Files that should be ignored by git. Add seperate .gitignore files in sub folders if needed
│                               
├── environment.yml                     <- Conda environment definition for ensuring consistent setup across environments
├── LICENSE
├── README.md                           <- The top-level README for developers using this project.
├── requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g.
│                                          Might not be needed if using environment.yml.
│
├── CODE
│   ├── RUN_PIPELINE.py                 <- Main script. Runs data acquisition, preprocessing, model training, generating and saving forecasts. Has configurations (explained below)
│   ├── GET_YAHOO_DATA.py               <- Contains functions to call yahoofinancials package to acquire data. Called within RUN_PIPELINE.py
│   ├── RUN_MODEL.py                    <- Contains functions to clean & transform data, generate features, train and select model, and generate and save forecasts. Called within RUN_PIPELINE.py
│   ├── CONVERT_FORECASTS_TO_JSON.py    <- Converts saved forecasts to json, for consumption in Flask. Called within RUN_PIPELINE.py
│   ├── GET_FORECASTS.py                <- Script for Flask app, to provide an endpoint to query forecasts
│   ├── Multi_1_to_90_days_DEV.ipynb    <- Jupyter notebook for model development. Useful for visualising data trends, training and validation losses, model tuning, features selection. Important for development, not to be used in production setting.
│   └── EDA_pt2.ipynb                   <- Jupyter notebook for exploratory data analysis. Contains key insights around concept drift, data detrending etc.
│
├── DATA
│   ├── List of Tickers.xlsx            <- Contains list of ticker symbols. Required file.
│   ├── RAW_DATA_FOR_EDA.xlsx           <- Raw data used for EDA. Contains data from 1st Nov 2011 to 31st October 2020.
│   └── RAW_DATA.csv                    <- Raw data used for modelling, generated by GET_YAHOO_DATA.py when run within RUN_PIPELINE.py
|
├── RESULTS                   
│   ├── saved_forecasts_PRD.csv         <- Saved forecasts. Regenerated each time RUN_PIPELINE.py is executed.
│   ├── saved_forecasts_PRD.json        <- Saved forecasts in JSON format. Used in Flask endpoint. Regenerated each time RUN_PIPELINE.py is executed.
│   ├── saved_descaled_data.csv         <- Saved copy of processed data, with min-max scaling removed. Used for diagnosing model performance. Regenerated each time RUN_PIPELINE.py is executed.
│   └── saved_scaled_data.csv           <- Saved copy of processed data, with min-max scaling applied. Used for diagnosing model performance. Regenerated each time RUN_PIPELINE.py is executed.
│
├── IMAGES                              <- Images used in project wiki 
|
└── SAMPLE_RESULTS                      <- Example run of modelling script, with a holdout testset of 90 days, forecasing from t=0 at 14th July 2020 
    └── Multi_1_to_90_days_DEV.ipynb    <- Sample script with debug mode (visualisation of results) to exhibit results of model. RAW_DATA_TEST_PRD_SCRIPT
    ├── RAW_DATA_TEST_PRD_SCRIPT.csv    <- Sample data, up to 14th July 2020
    └── evaluation.xlsx                 <- Evaluation of results. MAPE of 10% across t=+1 to t=+90

```

## Setup guide
You can find a step-by-step guide to run this project [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/wiki/1.-Setup-guide).

## Pipeline explanation
A walkthrough of the start-to-end script logic is available [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/wiki/2.-Pipeline-explanation).

## Technical documentation
Docstrings of functions used can be found [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/wiki/5.-Technical-documentation).

## Exploratory data analysis
Key insights noted in our dataset are available [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/wiki/3.-Exploratory-data-analysis).

## Results
We have evaluated our model against a holdout test set of 90 days, forecasting from t=0 at 14th July 2020. Results are summarised [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/wiki/4.-Results).

## Moving forward
There are number of areas we can improve on

### 1. Overcome overfitting
As explained [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/wiki/4.-Results), we seem to be overfitting to the training set. In this first attempt, we did not spend much time tuning our models to handle this. We can consider tuning our decision trees better moving forward.

### 2. More data
We should ideally aim to obtain data around supply. This is possible be a key determinant for gold prices. We can likely identify supply of key exporting nations, and use that as a proxy of global supply.

### 3. Saving historical forecasts
At present, we overwrite our forecasts at each run. Ideally, we should saving our forecasts to a database, indexed by date of model run. We should also ideally store our raw data in a database for future use, instead of always running the query from 2011.

### 4. Investigate t=2000
As noted [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/wiki/3.-Exploratory-data-analysis), we see a shift in our relationsips at t=2000. We did not dive deeper into what happened here. However, looking into this may give us an insight into whether there has been a fundamental change in the relationships between our variables.


## Licensing
This project is licensed under the terms of the MIT license.


## References
* https://oilprice.com/Energy/Energy-General/The-Energy-Model-That-Can-Predict-Gold-Prices.html
* https://investorplace.com/2011/07/5-hard-asset-alternatives-to-gold/
* https://towardsdatascience.com/understanding-dataset-shift-f2a5a262a766

## Disclaimer
Please use this at your own risk. The purpose of this project is to showcase how machine learning can be used to forecast prices of an asset. It should not be construed as professional advice or guidance for trading, investments or business decisions. However, feel free to extend upon this idea with your own efforts.

