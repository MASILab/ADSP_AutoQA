from flask import Flask, render_template, send_file, url_for
import os
from pathlib import Path
import argparse

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

args = pa()

QA_directory_abs = args.QA_directory

#assert that the QA directory is an absolute path that exists
assert os.path.isabs(QA_directory_abs), "The QA directory path must be an absolute path."
assert os.path.exists(QA_directory_abs), "The QA directory path does not exist."

QA_directory = QA_directory_abs

# Change this to the absolute path where your PDFs are stored
# PDF_DIRECTORY = '/path/to/your/pdf_directory'

@app.route('/')
def index():
    PDF_DIRECTORY = '../qa_data/ADNI/SLANT/'
    pdf_files = [f for f in os.listdir(QA_directory) if f.endswith('.pdf')]
    return render_template('index_pdf.html', pdf_files=pdf_files, pdf_name=pdf_files[0])

#@app.route('/datasets/<path:clicked_path>/<path:pipeline>/<path:pdf_name>')
@app.route('/<path:pdf_name>')
def serve_pdf(pdf_name):
    pdf_path = os.path.join(QA_directory, pdf_name)
    if os.path.isfile(pdf_path):
        return send_file(pdf_path)
    else:
        return 'PDF not found', 404

if __name__ == '__main__':
    app.run(debug=True)
