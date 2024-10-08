from flask import Flask, render_template

app = Flask(__name__)

# Example: Reading a DataFrame from CSV (replace with your own data loading logic)
#df = pd.read_csv('/Users/Michael/CompSci/Vanderbilt/Classes/test/HTML/qa_data/ADNI/PreQual/QA.csv')

# Convert DataFrame to JSON
#df_json = df.to_json(orient='records')

@app.route('/')
def index():
    return render_template('brython.html')

if __name__ == '__main__':
    app.run(debug=True)
