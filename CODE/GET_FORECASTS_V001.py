"""
Flask app
"""

import pandas as pd
from flask import Flask
import json


app = Flask(__name__)

# runs in local host http://127.0.0.1:5000/latest/all
@app.route('/latest/<horizon>')
def weather(horizon):
    """
    Reads json and outputs based on selected paramter
    Horizon can be either "all" or an integer between 1 and 90 representing desired timestamp
    Eg: http://127.0.0.1:5000/latest/all or http://127.0.0.1:5000/latest/54

    Parameters
    ----------
    horizon : string
        Horizon can be either "all" or an integer between 1 and 90 representing desired timestamp

    Returns
    -------
    output
        Json to output to page
    """     
    with open(r'.\RESULTS\saved_forecasts_PRD.json', 'r') as jsonfile:
        file_data = json.loads(jsonfile.read())

    if horizon == "all":
        output = json.dumps(file_data)

    else:
        output = json.dumps(file_data[horizon]) 

    return output

# Get setup so that if we call the app directly (and it isn't being imported elsewhere)
if __name__ == '__main__':
    app.run(debug=True)
