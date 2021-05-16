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
from dataset_names import dataset_names
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from age_lims import age_lims
from sklearn.svm import LinearSVR
from downsample_fracs import downsample_fracs
from sklearn.metrics import r2_score


def run_model(model):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    dataset = "UKB"
    file_name = dataset_names[dataset]

    #Loop over age range and sample size files and run the model, save the predictions into new dataframe
    for i in age_lims["lower"][dataset]:
        for j in downsample_fracs[dataset]:

            data = pd.read_csv(f"../data/{file_name}_lower_age_{i}_frac_{j}.csv")

            #True age for the data
            y = data['Age']

            #MRI variables
            x = data.copy()

            x = x.drop("Age",1)
            x = x.drop("eid",1)
            x = x.drop("Sex_cat",1)

            # Scaling using inter-quartile range
            scaler = RobustScaler()
            x = scaler.fit_transform(x)

            # define the model
            if(model=="XGB"):
                #xg_reg = xgb.XGBRegressor(objective= 'reg:squarederror',nthread=4,seed=42)
                M = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = reg_params[dataset]['learning_rate'],
                max_depth = reg_params[dataset]['max_depth'], n_estimators = reg_params[dataset]['n_estimators'], verbose = True, random_state=42)

            if(model=="SVR"):

                M = LinearSVR(max_iter=10000,C=1.5) #default C = 1.5

            pred = cross_val_predict(M, x, y, cv=reg_params['n_cv'], n_jobs=reg_params[dataset]['n_jobs'])

            #Add predictions to the dataframe
            data['pred'] = pred

            data['BAG'] = data['pred'] - data['Age']

            r2 = r2_score(data['Age'],data['pred'])
            print("R2 = %s" % r2)

            #Linear correction for the predicted age
            z = np.polyfit(data['Age'], data['pred'], 1)

            data['pred_corr'] = data['Age'] + data['pred'] - (z[1] +z[0]*data['Age'])

            #Store the dataframe
            data.to_csv(f"../data/{file_name}_lower_age_{i}_frac_{j}_with_pred{model_name}.csv")


def main():
    parser = argparse.ArgumentParser(description='Run model on the data and store predictions')
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    args = parser.parse_args()

    run_model(args.Model)

if __name__ == '__main__':
    main()
