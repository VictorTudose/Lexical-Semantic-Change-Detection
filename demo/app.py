from flask import Flask, render_template, request
from os import listdir
import json
import pandas as pd
import plotly
import plotly.express as px
import json
import csv
import numpy as np

from sklearn.manifold import TSNE
from torch import embedding


app = Flask(__name__)

def get_embeddings(file_name, k = 40):
    keys = []
    embeddings = []

    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        sup = 0
        for row in csv_reader:
            keys.append(row[0])
            embeddings.append([row[1:]])
            sup += 1
            if sup == k:
                break
    tsne_model_3d = TSNE(n_components=3, learning_rate='auto', init='random')
    embeddings = np.array(embeddings)
    n, m, k = embeddings.shape
    embeddings_3d = np.array(tsne_model_3d.fit_transform(embeddings.reshape(n * m, k))).reshape(n, m, 3)
    embeddings_3d = [embedding[0] for embedding in embeddings_3d]
    return embeddings_3d, keys

@app.route('/results')
def results():
    args = request.args
    model = {}
    if 'results' in args:
        result_directory = args['results']
        model = json.load(open(f"data/{result_directory}/desc.json", 'r'))
        model["root"] = f"data/{result_directory}/"
        embeddings, keys = get_embeddings(f"data/{result_directory}/words.csv")
        df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'], index=keys)
        fig = px.scatter_3d(df, x='x', y='y', z='z')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('results.html',graphJSON=graphJSON, model = model)
    else:
        return render_template('results.html', model = {})

@app.route('/')
def index():
    results = []
    results_directories = listdir('data')
    for result_directory in results_directories:
        results.append(result_directory.replace('_', ' '))
    model = {"results": results}
    return render_template('index.html', model = model)

if __name__ == '__main__':
    app.run(debug=True)