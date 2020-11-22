# Forecasting-SPDR-Gold-Trust-ETF-Prices (PRD)
This project aims to forecast [SPDR Gold Trust ETF (NYSEARCA: GLD)](https://finance.yahoo.com/quote/GLD/profile?p=GLD) for t+1 to t+90. We then build a simple flask app to provide an endpoint to retrieve these forecasts.

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
├── .gitignore               <- Files that should be ignored by git. Add seperate .gitignore files in sub folders if 
│                               needed
├── environment.yml          <- Conda environment definition for ensuring consistent setup across environments
├── LICENSE
├── README.md                <- The top-level README for developers using this project.
├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
│                               Might not be needed if using environment.yml.
│
├── CODE
│   ├── RUN_PIPELINE.py      <- Main script. Runs data acquisition, preprocessing, model training, generating and saving forecasts. Has configurations (explained below)
│   ├── GET_YAHOO_DATA.py    <- Contains functions to call yahoofinancials package to acquire data. Called within RUN_PIPELINE.py
│   ├── RUN_MODEL.py         <- Contains functions to clean & transform data, generate features, train and select model, and generate and save forecasts. Called within RUN_PIPELINE.py
│   ├── CONVERT_FORECASTS_TO_JSON.py     <- Converts saved forecasts to json, for consumption in Flask. Called within RUN_PIPELINE.py
│   ├── GET_FORECASTS.py     <- Script for Flask app, to provide an endpoint to query forecasts.
│
├── DATA
│   ├── List of Tickers.xlsx <- Contains list of ticker symbols. Required file.
│   ├── RAW_DATA.csv         <- Raw data used for modelling, generated by GET_YAHOO_DATA.py when run within RUN_PIPELINE.py
|
├── docs                     <- Documentation
│   ├── data_science_code_of_conduct.md  <- Code of conduct.
│   ├── process_documentation.md         <- Standard template for documenting process and decisions.
│   └── writeup              <- Sphinx project for project writeup including auto generated API.
│      ├── conf.py           <- Sphinx configurtation file.
│      ├── index.rst         <- Start page.
│      ├── make.bat          <- For generating documentation (Windows)
│      └── Makefikle         <- For generating documentation (make)
│
├── examples                 <- Add folders as needed e.g. examples, eda, use case
│
├── extras                   <- Miscellaneous extras.
│   └── add_explorer_context_shortcuts.reg    <- Adds additional Windows Explorer context menus for starting jupyter.
│
├── notebooks                <- Notebooks for analysis and testing
│   ├── eda                  <- Notebooks for EDA
│   │   └── example.ipynb    <- Example python notebook
│   ├── features             <- Notebooks for generating and analysing features (1 per feature)
│   ├── modelling            <- Notebooks for modelling
│   └── preprocessing        <- Notebooks for Preprocessing 
│
├── scripts                  <- Standalone scripts
│   ├── deploy               <- MLOps scripts for deployment (WIP)
│   │   └── score.py         <- Scoring script
│   ├── train                <- MLOps scripts for training
│   │   ├── submit-train.py  <- Script for submitting a training run to Azure ML Service
│   │   ├── submit-train-local.py <- Script for local training using Azure ML
│   │   └── train.py         <- Example training script using the iris dataset
│   ├── example.py           <- Example sctipt
│   └── MLOps.ipynb          <- End to end MLOps example (To be refactored into the above)
│
├── src                      <- Code for use in this project.
│   └── examplepackage       <- Example python package - place shared code in such a package
│       ├── __init__.py      <- Python package initialisation
│       ├── examplemodule.py <- Example module with functions and naming / commenting best practices
│       ├── features.py      <- Feature engineering functionality
│       ├── io.py            <- IO functionality
│       └── pipeline.py      <- Pipeline functionality
│
└── tests                    <- Test cases (named after module)
    ├── test_notebook.py     <- Example testing that Jupyter notebooks run without errors
    ├── examplepackage       <- examplepackage tests
        ├── examplemodule    <- examplemodule tests (1 file per method tested)
        ├── features         <- features tests
        ├── io               <- io tests
        └── pipeline         <- pipeline tests
```

## References
* https://oilprice.com/Energy/Energy-General/The-Energy-Model-That-Can-Predict-Gold-Prices.html
* https://investorplace.com/2011/07/5-hard-asset-alternatives-to-gold/

## Disclaimer
Please use this at your own risk. The purpose of this project is to showcase how machine learning can be used. It should not be used for professional or business decision-making.

