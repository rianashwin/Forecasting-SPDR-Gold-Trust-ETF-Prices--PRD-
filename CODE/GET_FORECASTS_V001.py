import pandas as pd
from flask import Flask
import json


app = Flask(__name__)

# runs in local host http://127.0.0.1:5000/latest/all
@app.route('/latest/<horizon>')
def weather(horizon):
    
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
