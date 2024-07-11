from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    # Save the file in the uploads folder
    temp_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_file_path)
    df = pd.read_csv(temp_file_path)
    return jsonify({'data': df.to_dict(orient='records'), 'file_name': file.filename})

@app.route('/save', methods=['POST'])
def save_file():
    data = request.json.get('data')
    file_name = request.json.get('file_name')
    if not data or not file_name:
        return "Invalid data or file name", 400
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return "File saved successfully", 200
    except Exception as e:
        return str(e), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)



