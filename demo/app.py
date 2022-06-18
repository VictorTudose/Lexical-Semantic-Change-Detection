from flask import Flask, render_template, request
from os import listdir
import json

app = Flask(__name__)

@app.route('/results')
def index():
    args = request.args
    model = {}
    if 'results' in args:
        model = json.loads(f"data/{args['results']}/desc.json")

        return render_template('results.html', model = model)
    else:
        return render_template('results.html', model = {})

@app.route('/')
def index():
    results = []
    results_directories = listdir('data')
    for result_directory in results_directories:
        results.append(result_directory.replace('_', ' '))
    print(f'len(results): {len(results)}')
    model = {"results": results}
    return render_template('index.html', model = model)

if __name__ == '__main__':
    app.run(debug=True)

