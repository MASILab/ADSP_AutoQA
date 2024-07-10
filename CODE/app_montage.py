from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file
import pandas as pd
import os, json, io, argparse, re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import itertools

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


def get_BIDS_fields_from_png(filename, return_pipeline=False):
    """
    Given a QA png filename, return the BIDS fields.
    """
    #pattern = r"sub-(?P<sub>\d+)_ses-(?P<ses>\d+)_\w+acq-(?P<acq>\d+)run-(?P<run>\d+)\.png"
    pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(\w+))(?:(acq-\w+))?(?:(run-\d{1,2}))?.png'
    match = re.match(pattern, filename)
    assert match, f"Filename {filename} does not match the expected pattern."
    tags = {'sub': match.group(1), 'ses': match.group(2), 'acq': match.group(4), 'run': match.group(5)}
    if return_pipeline:
        tags['pipeline'] = match.group(3)
    return tags

def create_json_dict(filepaths):
    """
    Given a list of filenames, create the initial BIDS json dictionary
    """

    user = os.getlogin()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nested_d = {}
    for png in tqdm(filepaths):
        current_d = nested_d
        tags = get_BIDS_fields_from_png(png)
        sub, ses, acq, run = tags['sub'], tags['ses'], tags['acq'], tags['run']
        for tag in [sub, ses, acq, run]:
            if tag:
                current_d = current_d.setdefault(tag, {})
        #set the default values
        row = {'QA_status': 'yes', 'reason': '', 'user': user, 'date': date}
        current_d.update(row)
        current_d = nested_d

    return nested_d

    #print(json.dumps(nested_d, indent=4))

def get_tag_type(d):
    """
    Returns the type of BIDS tag for sub, ses, acq, run. Throws an error if the tag is not one of these.
    """
    tag_types = {
        'sub': 'sub',
        'ses': 'ses',
        'acq': 'acq',
        'run': 'run'
    }
    for key, value in tag_types.items():
        if d.startswith(key):
            return value
    assert False, f"Unknown tag type: {d}"

def get_leaf_dicts(d, path=None, curr_dict=None):
    """
    Given a nested json dictionary, return a list of the leaf dictionaries
    """
    if path is None:
        path = []
    if curr_dict is None:
        curr_dict = {}
    leaf_dicts = []
    for key, value in d.items():
        #print(key)
        if isinstance(value, dict):
            new_path = path + [key]
            curr_dict[get_tag_type(key)] = key  #### For some reason, curr_dict is carrying over previous values
            leaf_dicts.extend(get_leaf_dicts(value, new_path, curr_dict))
        else:
            leaf_dicts.append((path, d))
            break
    return leaf_dicts

def convert_json_to_csv(json_dict, pipeline_path):
    """
    Given a QA JSON dictionary, convert it to a CSV file
    """

    #get the leaf dictionaries
    leaf_dicts = get_leaf_dicts(json_dict)

    #make sure that the paths are unique and the dictionary has all the information
    for paths,ds in leaf_dicts:
        for path in paths:
            ds[path[:3]] = path
            assert path in ds.values(), f"Path {path} not in dict {ds}"
        if 'run' not in ds:
            ds['run'] = ''
        if 'acq' not in ds:
            ds['acq'] = ''
        if 'ses' not in ds:
            ds['ses'] = ''
    #now get a list of only the leaf dictionaries
    leaf_dicts = [ds for paths,ds in leaf_dicts]
    #finally, convert to a csv
    header = ['sub', 'ses', 'acq', 'run', 'QA_status', 'reason', 'user', 'date']
    df = pd.DataFrame(leaf_dicts)
    #reorder the columns accroding to the header
    df = df[header]
    #replace NaN with empty string
    df = df.fillna('')

    df.to_csv(pipeline_path / 'QA.csv', index=False)

    return df

def read_csv_to_json(df):
    """
    Given a QA CSV dataframe, convert it to a QA JSON dictionary
    """

    json_data = {}

    for index, row in df.iterrows():
        #sub, ses, acq, run = row['sub'], row['ses'], row['acq'], row['run']
        qa_status, reason, user, date = row['QA_status'], row['reason'], row['user'], row['date']
        current_d = json_data
        has_d = {}
        for tag in ['sub', 'ses', 'acq', 'run']:
            if row[tag]:
                current_d = current_d.setdefault(row[tag], {})
                has_d[tag] = row[tag]
        #set the values
        add_row = {'QA_status': qa_status, 'reason': reason, 'user': user, 'date': date}
        if 'run' not in has_d:
            add_row.update({'run': ''})
        if 'acq' not in has_d:
            add_row.update({'acq': ''})
        if 'ses' not in has_d:
            add_row.update({'ses': ''})
        add_row.update(has_d)
        current_d.update(add_row)
        current_d = json_data
    
    #print(json.dumps(json_data, indent=4))

    return json_data

def compare_dicts(d1, d2):
    """
    Compare two dictionaries
    """
    
    #assert len(d1) == len(d2), "Dictionaries have different lengths"
    for key in d1:
        #print(key)
        #print(d1)
        #print(d2)
        assert key in d2, f"Key {key} not in d2. d1: {d1} \n d2: {d2}"
        if isinstance(d1[key], dict):
            compare_dicts(d1[key], d2[key])
        else:
            assert d1[key] == d2[key], f"Values for key {key} are different: {d1[key]} vs {d2[key]}"

def are_unique_qa_dicts(dict_list):
    """
    Given a list of qa dictionaries, check that no two dictionaries are the same

    Only considers the sub, ses, acq, and run elements
    """

    def add_items(curr_set, elt):
        curr_set.add(elt)
        return len(curr_set)

    seen = set()
    for d in tqdm(dict_list):
        sub, ses, acq, run = d['sub'], d['ses'], d['acq'], d['run']
        if len(seen) == add_items(seen, (sub, ses, acq, run)):
            return False
    return True

def assert_tags_in_dict(paths, leaf_dicts):
    """
    For given lists of paths and leaf dictionaries, assert that the paths are in the dictionaries
    """
    for paths,ds in zip(paths, leaf_dicts):
        for path in paths:
            assert path in ds.values(), f"Path {path} not in dict {ds}"

def check_png_for_json(dicts, pngs):
    """
    Given a list of QA json leaf dictionaries and list of pngs, make sure that every single json entry has a corresponding png file
    """

    #get the pipeline
    pipeline = get_BIDS_fields_from_png(pngs[0], return_pipeline=True)['pipeline']

    for dic in dicts:
        sub, ses, acq, run = dic['sub'], dic['ses'], dic['acq'], dic['run']
        png = f'{sub}_'
        if ses:
            png += f"{ses}_"
        png += f"{pipeline}"
        if acq:
            png += f"{acq}"
        if run:
            png += f"{run}"
        png += ".png"
        assert png in pngs, f"PNG {png} from {dic} not in list of pngs"

def check_json_for_png(nested, pngs):
    """
    Given a nested json and list of pngs, make sure that every single png file has a corresponding json entry.

    If it does not, then add the default values to the json file.
    """

    user = os.getlogin()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for png in pngs:
        tags = get_BIDS_fields_from_png(png)
        sub, ses, acq, run = tags['sub'], tags['ses'], tags['acq'], tags['run']
        current_d = nested
        for tag in [sub, ses, acq, run]:
            if tag:
                try:
                    current_d = current_d[tag]
                except KeyError:
                    print(f"PNG {png} has no corresponding json entry. Adding to json file.")
                    current_d = current_d.setdefault(tag, {})
        #if current_d is blank, then we need to add the default values
        if not current_d:
            row = {'QA_status': 'yes', 'reason': '', 'user': os.getlogin(), 'date': date}
            current_d.update(row)
            current_d.update(tags)
    
    return nested

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

    #define the current user (obtained from os)
    user = os.getlogin()

    #get the current date and time as a string
    now = datetime.now()

    # Get the list of PNG files in the pipeline directory (or pdfs)
    pipeline_path = Path(QA_directory + '/' + clicked_path + '/' + pipeline)
    pngs = [str(x.relative_to(QA_directory)) for x in itertools.chain(pipeline_path.glob('*.pdf'), pipeline_path.glob('*.png'))]  # Convert paths to relative paths

    ### check to make sure that there are not both pngs and pdfs
    ext = pngs[0].split('.')[-1]
    assert all([ x.split('.')[-1] == ext for x in pngs]), "There are both pngs and pdfs in the pipeline directory. Please correct before attempting QA."

    #pass image paths to montage.html so they can be loaded
    image_paths = [str(png) for png in pngs]
    pngs_files = [ x.split('/')[-1] for x in pngs]
    #print("Image paths:", image_paths)

    #check to see if the json file exists. If it doesn't, create it
    json_path = pipeline_path / 'QA.json'
    if not json_path.exists():
        #create the json dictionary
        json_dict = create_json_dict(pngs_files)
        #convert the json dictionary to a csv
        df = convert_json_to_csv(json_dict, pipeline_path)
    
    #otherwise, read the json file
    else:
        #read the json file
        with open(json_path, 'r') as f:
            json_dict = json.load(f)

        #check to make sure there are no duplicate QA dictionaries
        paths, leaf_dicts = zip(*get_leaf_dicts(json_dict)) #unzips the tuples into their separate lists
        assert are_unique_qa_dicts(leaf_dicts), "There are duplicate QA dictionaries in the json file {}. Please correct before attempting QA.".format(json_path)

        #check to make sure that the paths to the json dictionaries are correct
        assert_tags_in_dict(paths, leaf_dicts)

        #check to make sure that every json entry has a corresponding png file (throw an error if not)
        pngs_files = [ x.split('/')[-1] for x in pngs]
        check_png_for_json(leaf_dicts, pngs_files)

        #if the png does not have a corresponding json entry, it needs to be added
        json_dict = check_json_for_png(json_dict, pngs_files)


    #pass the dataframe to montage.html as a json
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