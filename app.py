from flask import Flask, jsonify,  request, render_template
from sklearn.externals import joblib
from capstone_project_core_logic import recommendation_system
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        int_features = [x for x in request.form.values()]
        final_features = [np.array(int_features)]
        predicted_list = recommendation_system(final_features)
        return render_template('index.html', prediction_text='Churn Output {}'.format(predicted_list))
    else :
        return render_template('index.html')

@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(recommendation_system(data.values()))
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
