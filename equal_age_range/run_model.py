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
from sklearn_rvm import EMRVR
from dataset_names import dataset_names
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
from age_lims import age_lims

def run_model(dataset, model, agerange, age):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    #Load the data

    #All age values
    if(age=="all"):
        suff = ""
    #Specific upper age range
    else:
        suff = f"_{agerange}_age_{age}"

    dataset_name = dataset_names[dataset]
    file_name = f"../data/{dataset_name}{suff}"

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
        x = x.drop('Scanner_cat',1)


    # Scaling using inter-quartile range
    scaler = RobustScaler()
    x = scaler.fit_transform(x)

    # define the model
    if(model == "XGB"):

        #xg_reg = xgb.XGBRegressor(objective= 'reg:squarederror',nthread=4,seed=17)
        M = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = reg_params[dataset]['learning_rate'],
        max_depth = reg_params[dataset]['max_depth'], n_estimators = reg_params[dataset]['n_estimators'], verbose = True, random_state=42)



    if(model=="SVR"):


        M = LinearSVR(max_iter=10000,C=reg_params[dataset]['C'])


    # run cross val predict with search to get predicition for everyone
    pred = cross_val_predict(M, x, y, cv=reg_params[dataset]['n_cv'], n_jobs=reg_params[dataset]['n_jobs'])


    #Add predictions to the dataframe
    data['pred'] = pred
    data['BAG'] = data['pred'] - data['Age']

    #Linear correction for the predicted age
    z = np.polyfit(data['Age'], data['pred'], 1)

    data['pred_corr'] = data['Age'] + data['pred'] - (z[1] +z[0]*data['Age'])

    #Store the dataframe
    data.to_csv(f"{file_name}_with_pred{model_name}.csv")

def main():
    parser = argparse.ArgumentParser(description='Run model on the data and store predictions')
    parser.add_argument("--Data", required=True, choices=["CamCan","UKB"], help="Dataset")
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    parser.add_argument("--AgeRange", required=True, choices=["lower","upper"], help="Whether to move lower or upper age")
    args = parser.parse_args()

    #Run over the age ranges
    age_ranges = ["all"]
    for a in age_lims[args.AgeRange][args.Data]:
        age_ranges.append(str(a))
    for a in age_ranges:
        run_model(args.Data, args.Model, args.AgeRange, a)

if __name__ == '__main__':
    main()
