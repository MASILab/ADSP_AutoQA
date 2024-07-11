from flask import Flask, render_template
import pandas as pd
import json
import os

app = Flask(__name__)

# Example: Reading a json file in as a nested dictionary
with open('qa_data/ADNI/PreQual/QA.json') as f:
    data = json.load(f)


@app.route('/')
def index():
    return render_template('json_test.html', data=data, user=os.getlogin())

if __name__ == '__main__':
    app.run(debug=True)
