from __main__ import *
#######################################################
# Define functions
#######################################################


def convert_forecasts_to_json():
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
                    "Predicted SDPR Gold ETF Prices (USD)" : row["Predicted - Descaled"]
                }
            
            json.dump(final_data, jsonfile)
            
            jsonfile.write('\n')
  