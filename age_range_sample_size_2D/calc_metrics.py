import sys, os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from age_lims import age_lims
from dataset_names import dataset_names
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import json
from downsample_fracs import downsample_fracs

#Calculations for various metrics
def calc_r(true, pred):
    return pearsonr(true, pred)

def calc_r2(true, pred):
    return r2_score(true, pred)

def calc_rmse(true, pred):
    return mean_squared_error(true, pred, squared = False)

def calc_mae(true, pred):
    return mean_absolute_error(true, pred)

def calc_mape(true, pred):
    return mean_absolute_percentage_error(true, pred)

def calc_mape(true, pred):
    true, pred = np.array(true), np.array(pred)
    return np.mean(np.abs((true - pred) / true)) * 100

def calc_rse(true, pred):
    rse_numerator = np.sum((true - pred)**2)
    rse_denominator = np.sum((true - true.mean())**2)
    return rse_numerator / rse_denominator

def calc_rae(true, pred):
    rae_numerator = np.sum(np.abs(true - pred))
    rae_denominator = np.sum(np.abs(true - true.mean()))
    return rae_numerator / rae_denominator

def calc_all(true, pred):
    #Pred. vs true correlation and p-value
    r, r_p = calc_r(true, pred)

    #r2
    r2 = calc_r2(true, pred)

    #RMSE
    rmse = calc_rmse(true, pred)

    #MAE
    mae = calc_mae(true, pred)

    #RSE (relative squared error, how much the prediction errors differ from the standard deviation of the true age)
    rse = calc_rse(true, pred)

    #RSE (relative absolute error)
    rae = calc_rae(true, pred)

    return r, r2, rmse, mae, rse, rae


def calc_metrics(model):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    dataset = "UKB"
    file_name = dataset_names[dataset]

    #Loop over age range and sample size files and run the model, save the predictions into new dataframe
    for i in age_lims["lower"][dataset]:
        for j in downsample_fracs[dataset]:
            print("Lower age range: %s" % i)
            print("Sample fraction: %s" % j)

            data = pd.read_csv(f"../data/{file_name}_lower_age_{i}_frac_{j}_with_pred{model_name}.csv")
            #Store results to a dict in JSON
            metric_dict = {}

            #Uncorrected and corrected prediction metrics
            corr = ["","_corr"]
            for c in corr:

                #Default results
                r, r2, rmse, mae, rse, rae = calc_all(data['Age'],data[f'pred{c}'])

                metric_dict[f"r{c}"] = r
                #metric_dict[f"r_p{c}"] = r_p
                metric_dict[f"r2{c}"] = r2
                metric_dict[f"RMSE{c}"] = rmse
                metric_dict[f"MAE{c}"] = mae
                #metric_dict[f"MAPE{c}"] = mape
                metric_dict[f"RSE{c}"] = rse
                metric_dict[f"RAE{c}"] = rae

                #Bootstrapping to get errors for each metric
                bs_vals = {}
                for m in metric_dict:
                    bs_vals[m] = []
                for k in range(0,200):
                    data_bs = data.sample(frac=1, replace=True)
                    r, r2, rmse, mae, rse, rae = calc_all(data_bs['Age'],data_bs[f'pred{c}'])

                    bs_vals["r"].append(r)
                    bs_vals["r2"].append(r2)
                    bs_vals["RMSE"].append(rmse)
                    bs_vals["MAE"].append(mae)
                    bs_vals["RSE"].append(rse)
                    bs_vals["RAE"].append(rae)

                for m in ["r","r2","RMSE","MAE","RSE","RAE"]:
                    metric_dict[f"{m}{c}_err"] = np.std(bs_vals[m])

                #Size of dataset
                metric_dict["n"] = len(data)
                print (metric_dict["n"])
            with open(f"../output/{file_name}_lower_age_{i}_frac_{j}_metrics{model_name}.json","w") as f:
                json.dump(metric_dict, f)


def main():
    parser = argparse.ArgumentParser(description='Calculate performance metrics for a given dataset')
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    args = parser.parse_args()

    calc_metrics(args.Model)

if __name__ == '__main__':
    main()
