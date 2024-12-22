import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import yaml
import os

#Load params

params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path,output_path,random_state,eval_x,eval_y):
    data = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.to_csv(output_path,index=False)

    #Eval data files creation
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=random_state)

    #Eval features file save
    os.makedirs(os.path.dirname(eval_x),exist_ok=True)
    X_test.to_csv(eval_x,index=False)

    #Eval pred file save
    os.makedirs(os.path.dirname(eval_y),exist_ok=True)
    y_test.to_csv(eval_y,index=False)

if __name__ =="__main__":
    preprocess(params['input'],params['output'],params['random_state'],params['X'],params['y'])