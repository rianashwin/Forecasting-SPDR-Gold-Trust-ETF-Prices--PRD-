# Forecasting-SPDR-Gold-Trust-ETF-Prices (PRD)
This project aims to forecast [SPDR Gold Trust ETF (NYSEARCA: GLD)](https://finance.yahoo.com/quote/GLD/profile?p=GLD) for t+1 to t+90. We then build a simple flask app to provide an endpoint to retrieve these forecasts.

## Background
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

## Requirements
There are a number of key packages used. Aside from the usual suspects of pandas, matplotlib, numpy, sklearn and keras, we need to use a python module that allows us to query data from Yahoo! Financials. Further notes for this package is available via [PyPI](https://pypi.org/project/yahoofinancials/). We can install this by running the following

    python -m pip install yahoofinancials
    
List of packages can be found [here](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/blob/main/requirements.txt). Installation can be completed using conda, as follows:
    
    conda create --name <env> --file <this file>
    
You can also create this environment from the [environment.yml](https://github.com/rianashwin/Forecasting-SPDR-Gold-Trust-ETF-Prices--PRD-/blob/main/environment.yml) file by running the following:

    conda env create -f environment.yml

## References
* https://oilprice.com/Energy/Energy-General/The-Energy-Model-That-Can-Predict-Gold-Prices.html
* https://investorplace.com/2011/07/5-hard-asset-alternatives-to-gold/

## Disclaimer
Please use this at your own risk. The purpose of this project is to showcase how machine learning can be used. It should not be used for professional or business decision-making.

