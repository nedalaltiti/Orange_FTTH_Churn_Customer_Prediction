import pickle
import orange_project
import pandas as pd
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    eva = ""
    test_file = pd.read_csv("test.csv")
    test = orange_project.preprocessing(test_file)
    X_test, y_test = test.drop('TARGET', axis=1), test['TARGET']


    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        y_pred = model.predict(X_test)
        eva = orange_project.evaluation(y_true=y_test,y_pred=y_pred)
        return eva


if __name__ == "__main__":
    app.run()




