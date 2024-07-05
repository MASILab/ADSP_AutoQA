from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file
import pandas as pd
import os
import io
import argparse
from pathlib import Path

def pa():
    parser = argparse.ArgumentParser(description="""

    Given the path to the QA directory, will set up a montage of images to QA.
                                     
    Depending on the image quality, the user can update the QA status and add a reason for the update.
                                
    The updated CSV will be saved to the specified save path as updates are made.

""")
    parser.add_argument('QA_directory', type=str, help='path to QA directory')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    return parser.parse_args()

app = Flask(__name__)
app.secret_key = 'supersecretkey'

args = pa()

QA_directory_abs = args.QA_directory

#assert that the QA directory is an absolute path that exists
assert os.path.isabs(QA_directory_abs), "The QA directory path must be an absolute path."
assert os.path.exists(QA_directory_abs), "The QA directory path does not exist."

QA_directory = QA_directory_abs


@app.route('/')
def index():
    datasets = [ x for x in Path(QA_directory).glob('*') if x.is_dir() ]
    datasets = sorted(datasets, key=lambda x: x.name)
    datasets = [ x.name for x in datasets ]
    if args.debug:
        print("Datasets:", datasets)
    return render_template('root.html', datasets=datasets)

@app.route('/datasets', methods=['POST'])
def load_datasets():
    data = request.get_json()
    path = data.get('path')

    # Here you can customize what datasets to show for the selected path
    # For simplicity, I'll assume you fetch directories in a similar manner
    datasets = [str(x.name) for x in Path(QA_directory + '/' + path).glob('*') if x.is_dir()]

    return jsonify(datasets=datasets)

@app.route('/datasets/<path:clicked_path>')
def datasets(clicked_path):
    if args.debug:
        print("Clicked path:", clicked_path)
    datasets = [str(x.name) for x in Path(QA_directory + '/' + clicked_path).glob('*') if x.is_dir()]
    return render_template('datasets.html', clicked_path=clicked_path, directories=datasets)

@app.route('/datasets/<path:clicked_path>/<path:pipeline>')
def render_montage(clicked_path, pipeline):
    # Get the list of PNG files in the pipeline directory
    pipeline_path = Path(QA_directory + '/' + clicked_path + '/' + pipeline)
    pngs = [str(x.relative_to(QA_directory)) for x in pipeline_path.glob('*.png')]  # Convert paths to relative paths
    #print("PNGs:", pngs)
    # Construct URLs to serve these images
    #image_paths = [url_for('serve_file', filename=str(png)) for png in pngs]
    image_paths = [str(png) for png in pngs]
    #image_names = [str(x.name) for x in pipeline_path.glob('*.png')]
    print("Image paths:", image_paths)

    return render_template('montage.html', clicked_path=clicked_path, pipeline=pipeline, image_paths=image_paths)

#may need to create separate ones for PreQual, or others that use PDFs

@app.route('/datasets/<path:clicked_path>/<path:pipeline>/<path:image_filename>')
def serve_image(clicked_path, pipeline, image_filename):
    # Construct the full path to the image file
    image_path = os.path.join(QA_directory, clicked_path, pipeline, image_filename)

    # Check if the image file exists
    print("Checking for file:", image_path)
    if os.path.isfile(image_path):
        # Send the image file as a response
        print("Sending file:", image_path)
        return send_file(image_path, mimetype='image/png')
    else:
        # Return a 404 error if the file doesn't exist
        return 'Image not found', 404
# def serve_image(image_path):
#     # Construct the full path to the image file
#     #image_path = os.path.join(QA_directory, clicked_path, pipeline, image_filename)

#     # Check if the image file exists
#     if os.path.isfile(image_path):
#         # Send the image file as a response
#         return send_file(image_path, mimetype='image/png')  # Adjust mimetype as needed
#     else:
#         # Return a 404 error if the file doesn't exist
#         return 'Image not found', 404

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


if args.debug:
    app.run(debug=True)
else:
    app.run()