from flask import Flask, request, render_template, redirect, send_file
import pandas as pd
import os
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load', methods=['POST'])
def load_file():
    file_path = request.form['file_path']
    save_path = request.form['save_path']
    
    if not os.path.isfile(file_path):
        return "File not found. Please check the path and try again.", 400
    
    df = pd.read_csv(file_path)
    
    return render_template('edit.html', table=df.values.tolist(), columns=df.columns.tolist(), file_path=file_path, save_path=save_path)

@app.route('/save', methods=['POST'])
def save_file():
    edited_data = request.form['table-data']
    save_path = request.form['save-path']

    print("Save path:", save_path, '\n\n')

    df = pd.read_csv(io.StringIO(edited_data))

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df.to_csv(save_path, index=False)
    
    return f"File successfully saved to {save_path}"

if __name__ == '__main__':
    app.run(debug=True)
