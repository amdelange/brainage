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
from age_lims import age_lims, upper_age
from sklearn.model_selection import train_test_split

def run_model(dataset, model, agerange, age):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    #Load the full data, then split into A and B, then apply age cut
    dataset_name = dataset_names[dataset]
    file_name = f"../data/{dataset_name}"

    data = pd.read_csv(f"{file_name}.csv")

    data_A, data_B = train_test_split(data, test_size = 0.5, random_state=100)

    #Apply age cut to training data_A
    cut_direction = ""
    if(agerange == "lower"):
        cut_direction = ">="
    else:
        cut_direction = "<="
    print(f"Age cut: {cut_direction} {age}")

    #Check how many are in the narrowest age range, and use this to downsample
    min_age = age_lims[agerange][dataset][0]
    n_min = len(data_A.query(f"Age {cut_direction} {min_age}"))
    print("Number in training sample: %s" % n_min)

    data_A = data_A.query(f"Age {cut_direction} {age}")
    data_A = data_A.sample(n=int(n_min), replace=False, random_state=200)

    #Apply smallest age range cut to the test data
    data_B = data_B.query(f"Age {cut_direction} {min_age} and Age < {upper_age[dataset]}")
    print("Number in test sample: %s" % len(data_B))

    print(len(data_B))

    #True age for the data
    y = data_A['Age']

    #MRI variables
    x = data_A.copy()
    x_B = data_B.copy()

    if(dataset=="CamCan"):
        x = x.drop('Age',1)
        x = x.drop('ID',1)
        x = x.drop('Sex',1)

        x_B = x_B.drop('Age',1)
        x_B = x_B.drop('ID',1)
        x_B = x_B.drop('Sex',1)

    if(dataset=="UKB"):
        x = x.drop("Age",1)
        x = x.drop("eid",1)
        x = x.drop("Sex_cat",1)

        x_B = x_B.drop("Age",1)
        x_B = x_B.drop("eid",1)
        x_B = x_B.drop("Sex_cat",1)

    # Scaling using inter-quartile range
    scaler = RobustScaler()
    x = scaler.fit_transform(x)

    x_B = scaler.fit_transform(x_B)

    # define the model
    if(model == "XGB"):

        #xg_reg = xgb.XGBRegressor(objective= 'reg:squarederror',nthread=4,seed=17)
        M = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = reg_params[dataset]['learning_rate'],
        max_depth = reg_params[dataset]['max_depth'], n_estimators = reg_params[dataset]['n_estimators'], verbose = True, random_state=42)

    
    if(model=="SVR"):
        M = LinearSVR(max_iter=10000,C=1.5)

    M.fit(x, y)

    pred_B = M.predict(x_B)

    #Add predictions to the test dataframe
    data_B['pred'] = pred_B
    data_B['BAG'] = data_B['pred'] - data_B['Age']

    #Linear correction for the predicted age
    z = np.polyfit(data_B['Age'], data_B['pred'], 1)

    data_B['pred_corr'] = data_B['Age'] + data_B['pred'] - (z[1] +z[0]*data_B['Age'])

    #Store the dataframe
    #All age values
    if(age=="all"):
        suff = ""
    #Specific upper age range
    else:
        suff = f"_{agerange}_age_{age}"
    file_name = f"../data/{dataset_name}{suff}"

    data_B.to_csv(f"{file_name}_with_pred{model_name}_fixed_test_range.csv")

def main():
    parser = argparse.ArgumentParser(description='Run model on the data and store predictions on a fixed age range test set')
    parser.add_argument("--Data", required=True, choices=["CamCan","UKB"], help="Dataset")
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    parser.add_argument("--AgeRange", required=True, choices=["lower","upper"], help="Whether to move lower or upper age")
    args = parser.parse_args()

    #Run over the age ranges
    age_ranges = [] #["all"]
    for a in age_lims[args.AgeRange][args.Data]:
        age_ranges.append(str(a))
    for a in age_ranges:
        run_model(args.Data, args.Model, args.AgeRange, a)

if __name__ == '__main__':
    main()
