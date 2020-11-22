"""
Script to convert saved forecasts from csv to json
For ease of consumption by Flask app
"""

from __main__ import *
#######################################################
# Define functions
#######################################################


def convert_forecasts_to_json():
    """
    Ingests raw data, and outputs as json
    Json is indexed by horizon, and contains values for descaled predictions

    Parameters
    ----------
    None

    Returns
    -------
    None
    """ 
       
    fieldnames = (
        "Horizon","Actuals - Scaled","Actuals - Descaled","Predicted - Scaled","Predicted - Descaled", "Model Name"
    )

    with open(r'.\RESULTS\saved_forecasts_PRD.csv', 'r') as csvfile:
        with open(r'.\RESULTS\saved_forecasts_PRD.json', 'w') as jsonfile:
            
            next(csvfile)
            
            reader = csv.DictReader(csvfile, fieldnames)
            
            final_data = {}
            
            for row in reader:
                
                final_data[row["Horizon"]]= {
                    "Predicted SDPR Gold ETF Prices (USD)" : row["Predicted - Descaled"][1:-2]
                }
            
            json.dump(final_data, jsonfile)
            
            jsonfile.write('\n')
  