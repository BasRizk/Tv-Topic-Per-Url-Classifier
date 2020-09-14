# -*- coding: utf-8 -*-
from app import app
from flask import render_template
from inference_model import predict, load_pickle
from dataset_builder import scrap_web_text
from data_preprocessor import count_vector
from flask import Response, request
import pandas as pd

import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/classify", methods=["POST"])
def classify_url():
    def valid_text(text):
        if webpage_text is None or len(webpage_text) <= 0:
            return False
        return True
    
    try:
        url = request.get_json(force=True)["url_to_classify"]
    except Exception:
        return Response("Wrong JSON Format - HINT: url_to_classify=YOUR_URL")

        
    # url = "http://" + "/".join([url, url_sub, url_sub2])
    print("Recevied %s" % url)
    
    outcome = {
        0 : "Link is 'NOT' Related",
        1 : "Link is related"
        }
    
    webpage_text = scrap_web_text(url)
    if not valid_text(webpage_text):
        return Response("No text in the webapp available, hence considered not related")
    data = [[url, webpage_text]]
    df = pd.DataFrame(data, columns=['link','text'])
    
    # df, count_vectorizer, tf_idf_vectors = preprocessing(df, save_vocab=False)
    
    feature_names = load_pickle("feature.pkl")
    loaded_cv, tf_idf_vectors = count_vector(df, vocab_pickle=feature_names)
    
    X_test = pd.DataFrame(tf_idf_vectors.todense(), columns=feature_names)
    clf = load_pickle("model.pkl")
    
    y_test = predict(clf, X_test, verbose=False)

    return Response(outcome[y_test[0]])




# =============================================================================
# Debuging
# =============================================================================
@app.route('/debug')
def index():
    return plot_png()

def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig
