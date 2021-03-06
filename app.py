from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('salary.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['exp']
    arr  =np.array([[data1]])
    pred = model.predict(arr)
    return render_template('home.html', salary=pred)

if __name__ == "__main__":
    app.run(debug=True)