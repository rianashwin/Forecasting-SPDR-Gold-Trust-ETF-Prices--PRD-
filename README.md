# Forecasting-SPDR-Gold-Trust-ETF-Prices (PRD)
This project aims to forecast [SPDR Gold Trust ETF (NYSEARCA: GLD)](https://finance.yahoo.com/quote/GLD/profile?p=GLD) for t+1 to t+90. We then build a simple flask app to provide an endpoint to retrieve these forecasts.

## Background
Gold is typically seen as a safe-haven asset. Aside from being used for jewellery, it is also a reliable store of value. Typically, it hedges against the performance of bonds and stocks.

When building this model, we include features around alternative metals to gold (copper, palladium), as well as alternative assets such as crude oil. We also measure the strength of the stock market by including S&P 500 and the Russell 2000 index, as well as a measure of the strenght of the  USD, Japanese Yen and Swiss Francs.

Data sourced from [Yahoo! Finance](https://finance.yahoo.com/) using an API (explained in subsequent sections of this README):

| Symbol                 | Description | URL | URL | 
| :---                    | --- | --- | |
| GLD            | SPDR_Gold_Shares | | https://finance.yahoo.com/quote/GLD/profile?p=GLD |
| project_description     | A description to include in the README.md |  |link |
| repo_name               | The name of the github repository where the project will be held | link |
| conda_name              | The name of the conda environment to use |  |link |
| package_name            | A name for the generated python package. |  |link | 
| mlops_name              | Default name for Azure ML. | link | 
| mlops_compute_name      | Default Azure ML compute cluster name to use. |  |link | 
| author                  | The main author of the solution. Included in the setup.py file |  |link | 
| open_source_license     | What type of open source license the project will be released under |  |link | 
| devops_organisation     | An Azure DevOps organisation. Leave blank if you aren't using Azure DevOps |  |link | 

## References
* https://oilprice.com/Energy/Energy-General/The-Energy-Model-That-Can-Predict-Gold-Prices.html
* https://investorplace.com/2011/07/5-hard-asset-alternatives-to-gold/

## Disclaimer
Please use this at your own risk. The purpose of this project is to showcase how machine learning can be used. It should not be used for professional or business decision-making.

