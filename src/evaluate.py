import pandas as pd
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import mlflow
import os
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/govinbimsara/mlpiplines.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "govinbimsara"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "621db5d1a94907c87dc15c762a8057ddd7212d8b"

#Load params from yaml
params = yaml.safe_load(open("params.yaml"))['eval']

def evaluate(X,y,model_path):
    X_test = pd.read_csv(X)
    y_test = pd.read_csv(y)

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    model = pickle.load(open(model_path,'rb'))

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("f1",f1)
    mlflow.log_text(str(cm),"confusion_matrix.txt")
    mlflow.log_text(cr,"classification_report.txt")

if __name__ =="__main__":
    evaluate(params['X'],params['y'],params['model'])