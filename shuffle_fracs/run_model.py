import sys, os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import xgboost as xgb
from reg_params import reg_params
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from dataset_names import dataset_names
from sklearn.preprocessing import RobustScaler
from shuffle_fracs import shuffle_fracs
from sklearn.model_selection import train_test_split

def run_model(dataset,model,shuffle_frac):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"


    shuffle_f = 0.000001
    if(shuffle_frac!="none"):
        shuffle_f = float(shuffle_frac)

    #Load the data
    dataset_name = dataset_names[dataset]
    file_name = f"../data/{dataset_name}"

    data = pd.read_csv(f"{file_name}.csv")

    #True age for the data
    y = data['Age']

    #MRI variables
    x = data.copy()

    if(dataset=="CamCan"):
        x = x.drop('Age',1)
        x = x.drop('ID',1)
        x = x.drop('Sex',1)
    if(dataset=="UKB"):
        x = x.drop("Age",1)
        x = x.drop("eid",1)
        x = x.drop("Sex_cat",1)


    #Shuffle required subset of the data
    x_non_shuffle, x_shuffle, y_non_shuffle, y_shuffle = train_test_split(x, y, test_size = shuffle_f, random_state=10)
    #Shuffle age values
    y_shuffle = y_shuffle.sample(frac=1)

    x = x_non_shuffle.append(x_shuffle)
    y = y_non_shuffle.append(y_shuffle)

    # Scaling using inter-quartile range
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)

    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)


    # define the model
    if(model=="XGB"):
        # define the model
        #xg_reg = xgb.XGBRegressor(objective= 'reg:squarederror',nthread=4,seed=17)
        M = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = reg_params[dataset]['learning_rate'],
        max_depth = reg_params[dataset]['max_depth'], n_estimators = reg_params[dataset]['n_estimators'], verbose = True, random_state=42)

    if(model=="SVR"):

        c_val = 1.5 #default C = 1.5
        n_iter = 10000
        M = LinearSVR(max_iter=n_iter,C=c_val)

    # run cross val predict with search to get predicition for everyone, with n_cv and n_jobs defined in the reg_params.py script
    #pred = cross_val_predict(search, x, y, cv=cv_outer, n_jobs=reg_params['n_jobs'])
    pred = cross_val_predict(M, x_scaled, y, cv=10, n_jobs=reg_params[dataset]['n_jobs'])


    #Add predictions to the dataframe
    data_out = x.copy()
    data_out['Age'] = y
    data_out['pred'] = pred
    data_out['BAG'] = data_out['pred'] - data_out['Age']

    #Linear correction for the predicted age
    z = np.polyfit(data_out['Age'], data_out['pred'], 1)

    data_out['pred_corr'] = data_out['Age'] + data_out['pred'] - (z[1] +z[0]*data_out['Age'])

    #Store the dataframe
    data_out.to_csv(f"{file_name}_shuffle_frac_{shuffle_frac}_with_pred{model_name}.csv")

def main():
    parser = argparse.ArgumentParser(description='Run model on the data with a subset shuffled and store predictions')
    parser.add_argument("--Data", required=True, choices=["CamCan","UKB"], help="Dataset")
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    args = parser.parse_args()

    #Run over different shuffle fractions
    shuffle_f = ["none"]
    for s in shuffle_fracs[args.Data]:
        shuffle_f.append(str(s))
    for s in shuffle_f:
        run_model(args.Data, args.Model, s)

if __name__ == '__main__':
    main()
