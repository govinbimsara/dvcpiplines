import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import mlflow
from mlflow.models import infer_signature
import os
from sklearn.model_selection import GridSearchCV,train_test_split
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/govinbimsara/mlpiplines.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "govinbimsara"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "621db5d1a94907c87dc15c762a8057ddd7212d8b"

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1)
    grid_search.fit(X_train,y_train)
    return grid_search

#Load params from yaml
params = yaml.safe_load(open("params.yaml"))['train']


def train(data_path,model_path,random_state):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # mlflow.set_experiment('RF Classifier')
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    #Start mlflow run
    with mlflow.start_run():

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=random_state)
        # signature = infer_signature(X_train,y_train)

        param_grid = {
            'n_estimators' : params['n_estimators'],
            'max_depth' : params['max_depth'],
            'min_samples_split' : params['min_samples_split'],
            'min_samples_leaf' : params['min_samples_leaf'],
        }

        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

        best_model = grid_search.best_estimator_
        signature = infer_signature(X_test,best_model.predict(X_test))

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        cm = confusion_matrix(y_test,y_pred)
        cr = classification_report(y_test,y_pred)

        print(f'Accuracy: {accuracy}, f1: {f1}')

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("f1",f1)
        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")
        mlflow.log_text(str(params),"chosen_hyperparameters.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(
                sk_model = best_model,
                artifact_path = 'rf_model',
                input_example = X_train,
                signature = signature,
                registered_model_name = 'RF Classifier'
            )
        else:
            mlflow.sklearn.log_model(
            sk_model = best_model,
            artifact_path = 'rf_model',
            input_example = X_train,
            signature = signature
        )

        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))

        print(f'Model saved to {model_path}')

# if __name__ =="__main__":
#     train(params['input'],params['model'],params['random_state'])
