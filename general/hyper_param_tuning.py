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
from sklearn.model_selection import cross_val_score

def run_model(dataset, model):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    #Load the data (10% for use specifically in hyperparam tuning)
    dataset_name = dataset_names[dataset]
    file_name = f"../data/{dataset_name}_hyperparam_tuning"

    data = pd.read_csv(f"{file_name}.csv")
    print ('N in datafile:')
    print (len(data))


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


    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)

    # Scaling using inter-quartile range
    scaler = RobustScaler()
    x = scaler.fit_transform(x)

    results_file=savepath+'Output_%s_%s.txt'% (dataset,model)

    # define the model
    if(model == "XGB"):

        #xg_reg = xgb.XGBRegressor(objective= 'reg:squarederror',nthread=4,seed=17)
        M = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, verbose = True, random_state=42)

        # define search space
        parameters = {'max_depth': range(1, 11, 2),
                      'n_estimators': range(50, 400, 50),
                      'learning_rate': [0.001, 0.01, 0.1, 0.2]}

    elif(model == "SVR"):

        M = LinearSVR(max_iter=10000,random_state=42)
        parameters = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}


    # define search
    search = RandomizedSearchCV(
        estimator=M,
        param_distributions=parameters,
        scoring = 'neg_root_mean_squared_error',
        n_jobs = 4,
        cv = cv_inner,
        refit=True)


    with open(results_file, 'w') as text_file:

        text_file.write ('validating model\n')

        text_file.write ('------------------------------\n')
        text_file.write ('RMSE values:\n')
        RMSE = cross_val_score(search, x, y, cv=cv_outer,scoring='neg_root_mean_squared_error',n_jobs = 4)
        text_file.write('Mean and STD for RMSE: %.3f (%.3f)\n' % (np.mean(RMSE), np.std(RMSE)))

        text_file.write ('------------------------------\n')
        text_file.write ('MAE values:\n')
        MAE = cross_val_score(search, x, y, cv=cv_outer,scoring='neg_mean_absolute_error',n_jobs = 4)
        text_file.write('Mean and STD for MAE: %.3f (%.3f)\n' % (np.mean(MAE), np.std(MAE)))

        text_file.write ('------------------------------\n')
        text_file.write ('R2 values:\n')
        R2 = cross_val_score(search, x, y, cv=cv_outer,scoring='r2',n_jobs = 4)
        text_file.write('Mean and STD for R2: %.3f (%.3f)\n' % (np.mean(R2), np.std(R2)))

        #Get the best esimators from the search, and use those hyper params in a final model (which we will then use for feature importance)
        result = search.fit(x, y)
        # get the best performing model fit
        best_model = result.best_estimator_
        text_file.write(f'BEST MODEL: {best_model}\n')

        best_params = result.best_params_
        text_file.write(f'BEST PARAMS: {best_params}\n')


def main():
    parser = argparse.ArgumentParser(description='Run model on the data and store predictions')
    parser.add_argument("--Data", required=False, choices=["CamCan","UKB"], help="Dataset", default="UKB")
    parser.add_argument("--Model", required=True, choices=["XGB","SVR"], help="Model")
    args = parser.parse_args()

    run_model(args.Data, args.Model)

if __name__ == '__main__':
    main()
