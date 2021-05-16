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
from shuffle_fracs import shuffle_fracs

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

#Calculations for various metrics
def calc_r(true, pred):
    return pearsonr(true, pred)

def calc_r2(true, pred):
    return r2_score(true, pred)

def calc_rmse(true, pred):
    return mean_squared_error(true, pred, squared = False)

def calc_mae(true, pred):
    return mean_absolute_error(true, pred)

def calc_all(true, pred):
    #Pred. vs true correlation and p-value
    r, r_p = calc_r(true, pred)

    #r2
    r2 = calc_r2(true, pred)

    #RMSE
    rmse = calc_rmse(true, pred)

    #MAE
    mae = calc_mae(true, pred)

    return r, r2, rmse, mae


def calc_metrics(dataset, poly_order, model, shuffle_frac):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"


    #Load dataset with predictions
    suff = f"_shuffle_frac_{shuffle_frac}"

    dataset_name = dataset_names[dataset]
    file_name = f"../data/{dataset_name}{suff}_with_pred{model_name}"

    data = pd.read_csv(f"{file_name}.csv")

    #Calculate age correction via fit
    z = np.polyfit(data['Age'], data['pred'], poly_order)
    if(poly_order==1):
        data['pred_corr'] = data['Age'] + data['pred'] - (z[1] + z[0]*data['Age'])
    if(poly_order==2):
        data['pred_corr'] = data['Age'] + data['pred'] - (z[2] + z[1]*data['Age'] +z[0]*data['Age']**2)

    #Store results to a dict in JSON
    metric_dict = {}

    #Uncorrected and corrected prediction metrics
    corr = ["","_corr"]
    for c in corr:

        #################
        #Plot the predicted vs. true
        fig, ax = plt.subplots(figsize=(8,8))

        plt.scatter(data['Age'],data[f'pred{c}'],s=4,color='#2166ac')

        vmin = data['Age'].min()*0.95
        vmax = data['Age'].max()*1.05

        plt.xlim(vmin,vmax)
        plt.ylim(vmin,vmax)

        plt.xlabel("True age",fontsize=65,labelpad=10)
        if(c==""):
            prefix = "Pred. "
        else:
            prefix = "Corr. Pred. "
        plt.ylabel(f"{prefix}age",fontsize=65,labelpad=12)

        xvals = np.linspace(vmin, vmax, 1000)
        plt.plot(xvals,xvals,color='k',linestyle='--',alpha=0.5,linewidth=2)
        if(c==""):
            if(poly_order==1):
                fit_vals = z[1] + z[0]*xvals
            if(poly_order==2):
                fit_vals = z[2] + z[1]*xvals + z[0]*xvals**2
            plt.plot(xvals,fit_vals,color='#d6604d',linewidth=3.5)
        if(shuffle_frac=="none"):
            shuffle_frac_title = 0
        else:
            shuffle_frac_title = shuffle_frac
        plt.title(f"SF = {shuffle_frac_title}",fontsize=60, pad=12)

        ax.tick_params(axis='both', which='major', labelsize=50)

        margins = {"left":0.30,
                   "bottom":0.22,
                   "right":0.97,
                   "top":0.87
                    }
        fig.subplots_adjust(**margins)
        fig.savefig(f'../plots/True_vs_Pred_Age_SF_{dataset}_{shuffle_frac}{c}_poly_{poly_order}{model_name}.jpg')



        #Default results
        r, r2, rmse, mae = calc_all(data['Age'],data[f'pred{c}'])

        metric_dict[f"r{c}"] = r
        metric_dict[f"r2{c}"] = r2
        metric_dict[f"RMSE{c}"] = rmse
        metric_dict[f"MAE{c}"] = mae

        #Bootstrapping to get errors for each metric
        bs_vals = {}
        for m in metric_dict:
            bs_vals[m] = []
        for i in range(0,200):
            data_bs = data.sample(frac=1, replace=True)
            r, r2, rmse, mae = calc_all(data_bs['Age'],data_bs[f'pred{c}'])

            bs_vals["r"].append(r)
            bs_vals["r2"].append(r2)
            bs_vals["RMSE"].append(rmse)
            bs_vals["MAE"].append(mae)

        for m in ["r","r2","RMSE","MAE"]:
            metric_dict[f"{m}{c}_err"] = np.std(bs_vals[m])

        #Size of dataset
        metric_dict["n"] = len(data)

    with open(f"../output/{dataset_name}{suff}_metrics{model_name}.json","w") as f:
        json.dump(metric_dict, f)

def main():
    parser = argparse.ArgumentParser(description='Calculate performance metrics for a given dataset')
    parser.add_argument("--Data", required=True, choices=["CamCan","UKB"], help="Dataset")
    parser.add_argument("--Poly", required=True, choices=["1","2"], help="Order of polynomial for age correction")
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    args = parser.parse_args()

    #Run over different shuffle fractions
    shuffle_f = ["none"]
    for s in shuffle_fracs[args.Data]:
        shuffle_f.append(str(s))
    for s in shuffle_f:
        calc_metrics(args.Data, int(args.Poly),args.Model, s)

if __name__ == '__main__':
    main()
