from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import os
import io
import argparse

def pa():
    parser = argparse.ArgumentParser(description="""

    Given the path to the QA directory, will set up a montage of images to QA.
                                     
    Depending on the image quality, the user can update the QA status and add a reason for the update.
                                
    The updated CSV will be saved to the specified save path as updates are made.

""")
    parser.add_argument('QA_directory', type=str, help='path to QA directory')

    return parser.parse_args()

app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/load', methods=['POST'])
def load_file():
    file_path = request.form['file_path']
    save_path = request.form['save_path']
    
    if not os.path.isfile(file_path):
        return "File not found. Please check the path and try again.", 400
    
    df = pd.read_csv(file_path)
    
    return render_template('edit3.html', table=df.to_dict(orient='records'), columns=df.columns.tolist(), file_path=file_path, save_path=save_path)

@app.route('/update', methods=['POST'])
def update_file():
    data = request.form
    save_path = data['save_path']

    # Retrieve the form data
    sub = data['sub']
    ses = data['ses']
    acq = data['acq']
    run = data['run']
    new_qa_status = data['qa_status']
    new_reason = data['reason']

    # Load the original CSV
    df = pd.read_csv(data['file_path'])
    
    # Find the row to update
    row_index = df[(df['sub'] == sub) & (df['ses'] == ses) & (df['acq'] == acq) & (df['run'] == run)].index
    
    if not row_index.empty:
        index = row_index[0]
        original_qa_status = df.at[index, 'QA_status']
        original_reason = df.at[index, 'reason']

        # Update the values only if they are different
        if new_qa_status != original_qa_status or new_reason != original_reason:
            df.at[index, 'QA_status'] = new_qa_status
            df.at[index, 'reason'] = new_reason
            
            # Ensure the save directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the updated CSV
            df.to_csv(save_path, index=False)
            flash(f"File successfully saved to {save_path}", 'success')
        else:
            flash("No changes detected, nothing was saved.", 'info')
    else:
        flash("No matching row found to update.", 'error')
    
    return redirect(url_for('index'))

if __name__ == '__main__':

    args = pa()

    app.run(debug=True)
