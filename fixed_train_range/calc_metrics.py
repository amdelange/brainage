import sys, os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from age_lims import age_lims, upper_age, lower_age
from dataset_names import dataset_names
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from scipy.stats import pearsonr
import json

#Calculations for various metrics
def calc_r(true, pred):
    return pearsonr(true, pred)

def calc_r2(true, pred):
    return r2_score(true, pred)

def calc_rmse(true, pred):
    return mean_squared_error(true, pred, squared = False)

def calc_mae(true, pred):
    return mean_absolute_error(true, pred)

def calc_median_ae(true, pred):
    return median_absolute_error(true, pred)

def calc_wmae(true, pred, dataset, agerange, age):
    if(agerange=="lower"):
        #Get upper age
        x_age = upper_age[dataset]
    else:
        #Get lower age
        x_age = lower_age[dataset]
    if(age!="all"):
        if(agerange=="lower"):
            age_range = float(int(x_age) - int(age))
        else:
            age_range = float(int(age) -int(x_age))
    else:
        u_age = upper_age[dataset]
        l_age = lower_age[dataset]
        age_range = float(int(u_age) - int(l_age))

    return mean_absolute_error(true, pred)/age_range

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

def calc_all(true, pred, dataset, agerange, age):
    #Pred. vs true correlation and p-value
    r, r_p = calc_r(true, pred)

    #r2
    r2 = calc_r2(true, pred)

    #RMSE
    rmse = calc_rmse(true, pred)

    #MAE
    mae = calc_mae(true, pred)

    #Median AE
    median_ae = calc_median_ae(true, pred)

    #MAPE (mean absolute percentage error)
    #mape = calc_mape(true, pred)

    #RSE (relative squared error, how much the prediction errors differ from the standard deviation of the true age)
    rse = calc_rse(true, pred)

    #RSE (relative absolute error)
    rae = calc_rae(true, pred)

    #MAE divided by age range
    wmae = calc_wmae(true, pred, dataset, agerange, age)

    return r, r2, rmse, mae, median_ae, rse, rae, wmae


def calc_metrics(dataset, model, agerange, age):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    #Load dataset with predictions

    suff = f"_{agerange}_age_{age}"
    print(suff)
    dataset_name = dataset_names[dataset]
    file_name = f"../data/{dataset_name}{suff}_with_pred{model_name}_fixed_train_range"

    data = pd.read_csv(f"{file_name}.csv")
    mean_age = np.mean(data['Age'])
    std_age = np.std(data['Age'])
    mean_bag = np.mean(data['BAG'])
    std_bag = np.std(data['BAG'])
    data['BAG_corr'] = (data['pred_corr'] - data['Age'])

    mean_bag_corr = np.mean(data['BAG_corr'])
    std_bag_corr = np.std(data['BAG_corr'])
    print(f"Mean age ± STD : {mean_age} ± {std_age}")
    print(f"Mean BAG ± STD : {mean_bag} ± {std_bag}")
    print(f"Mean corr BAG ± STD : {mean_bag_corr} ± {std_bag_corr}")


    #print (len(data))
    #plt.scatter(data['Age'],data['pred'])
    #xvals = np.linspace(data['Age'].min(),data['Age'].max(),1000)
    #plt.plot(xvals,xvals,color='red')
    #plt.show()

    #Store results to a dict in JSON
    metric_dict = {}

    #Uncorrected and corrected prediction metrics
    corr = ["","_corr"]
    for c in corr:

        #Default results
        r, r2, rmse, mae, median_ae, rse, rae, wmae = calc_all(data['Age'],data[f'pred{c}'], dataset, agerange, age)

        metric_dict[f"r{c}"] = r
        #metric_dict[f"r_p{c}"] = r_p
        metric_dict[f"r2{c}"] = r2
        metric_dict[f"RMSE{c}"] = rmse
        metric_dict[f"MAE{c}"] = mae
        metric_dict[f"MedianAE{c}"] = median_ae
        metric_dict[f"WMAE{c}"] = wmae
        #metric_dict[f"MAPE{c}"] = mape
        metric_dict[f"RSE{c}"] = rse
        metric_dict[f"RAE{c}"] = rae

        #Bootstrapping to get errors for each metric
        bs_vals = {}
        for m in metric_dict:
            bs_vals[m] = []
        for i in range(0,200):
            data_bs = data.sample(frac=1, replace=True)
            r, r2, rmse, mae, median_ae, rse, rae, wmae = calc_all(data_bs['Age'],data_bs[f'pred{c}'], dataset, agerange, age)

            bs_vals["r"].append(r)
            bs_vals["r2"].append(r2)
            bs_vals["RMSE"].append(rmse)
            bs_vals["MAE"].append(mae)
            bs_vals["MedianAE"].append(median_ae)
            bs_vals["WMAE"].append(mae)
            bs_vals["RSE"].append(rse)
            bs_vals["RAE"].append(rae)

        for m in ["r","r2","RMSE","MAE","MedianAE","RSE","RAE","WMAE"]:
            metric_dict[f"{m}{c}_err"] = np.std(bs_vals[m])

        #Size of dataset
        metric_dict["n"] = len(data)

    with open(f"../output/{dataset_name}{suff}_metrics{model_name}_fixed_train_range.json","w") as f:
        json.dump(metric_dict, f)

def main():
    parser = argparse.ArgumentParser(description='Calculate performance metrics for a given dataset')
    parser.add_argument("--Data", required=True, choices=["CamCan","UKB"], help="Dataset")
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    parser.add_argument("--AgeRange", required=True, choices=["lower","upper"], help="Whether to move lower or upper age")
    args = parser.parse_args()

    #Run over all the age ranges
    age_ranges = []#["all"]
    for a in age_lims[args.AgeRange][args.Data]:
        age_ranges.append(str(a))
    for a in age_ranges:
        calc_metrics(args.Data, args.Model, args.AgeRange, a)

if __name__ == '__main__':
    main()
