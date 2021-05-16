import sys, os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import xgboost as xgb
from reg_params import reg_params
from sklearn.model_selection import cross_val_predict
from dataset_names import dataset_names
from shuffle_fracs import shuffle_fracs
import json

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

metric_titles = {'r': "$r$",
                 'r2': "$R^2$",
                 'RMSE': "RMSE [years]",
                 'MAE': "MAE [years]",
                 #'MAPE': "MAPE",
                 #'RSE': "RSE",
                 #'RAE': "RAE"
                 }

dataset_title = {'CamCan': 'Cam-CAN',
              'UKB': 'UKB'
                 }


def plot(dataset, model):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    shuffle_f = ["none"]
    for s in shuffle_fracs[dataset]:
        shuffle_f.append(str(s))

    x_index = []
    x_labels = []

    for i in range(0, len(shuffle_fracs[dataset])):
        x_index.append(i)
        x_labels.append(f"{shuffle_fracs[dataset][i]}")
    x_labels.reverse()

    results_dict = {}
    for s in shuffle_f:
        suff = f"_shuffle_frac_{s}"

        file_name = f"../output/{dataset_names[dataset]}{suff}_metrics{model_name}.json"

        with open(file_name,"r") as f:
            results_dict[s] = json.load(f)

    #Loop over metrics
    fig, ax = plt.subplots(2,2, figsize=(20,18))
    ax = ax.ravel()

    ax_i = 0

    for m in metric_titles:

        y_vals = []
        y_errs = []
        y_vals_corr = []
        y_errs_corr = []


        for s in shuffle_fracs[dataset]:

            val = results_dict[str(s)][m]
            y_vals.append(val)
            err = results_dict[str(s)][f"{m}_err"]
            y_errs.append(err)

            val_corr = results_dict[str(s)][f"{m}_corr"]
            y_vals_corr.append(val_corr)
            err_corr = results_dict[str(s)][f"{m}_corr_err"]
            y_errs_corr.append(err_corr)

        lab = ""
        label = ""
        corr_label = ""
        if(ax_i==0):
            #lab = "data models (N = %.0f)" % results_dict[str(shuffle_fracs[dataset][-1])]["n"]
            lab = "data models (N = 41,285)"
            label = "Shuffled "+lab
            corr_label = "Corr. shuffled "+lab
        #Reverse values
        y_vals.reverse()
        y_vals_corr.reverse()

        #Plot uncorrected values
        ax[ax_i].errorbar(x_index,y_vals,yerr=y_errs,fmt="o",markersize=6,capsize=8,color="k",label=label)
        #Plot corrected values
        ax[ax_i].errorbar(x_index,y_vals_corr,yerr=y_errs_corr,fmt="o",markersize=6,capsize=8,color="crimson",label=corr_label)

        #Plot result of full model to compare
        line_lab = ""
        line_label = ""
        corr_line_label = ""
        if(ax_i==0):
            line_lab = "model (N = 41,285)"
            #line_lab = "model (N = %.0f)" % results_dict["none"]["n"]
            line_label = "Full "+line_lab
            corr_line_label = "Corr. full "+line_lab
        ax[ax_i].axhline(results_dict["none"][m],color="k",alpha=0.5,linestyle='--',linewidth=2.5,label=line_label)
        ax[ax_i].axhspan(results_dict["none"][m]-results_dict["none"][f"{m}_err"], results_dict["none"][m]+results_dict["none"][f"{m}_err"], color="k", alpha=0.2)

        ax[ax_i].axhline(results_dict["none"][f"{m}_corr"],color="crimson",alpha=0.5,linestyle='--',linewidth=2.5,label=corr_line_label)
        ax[ax_i].axhspan(results_dict["none"][f"{m}_corr"]-results_dict["none"][f"{m}_corr_err"], results_dict["none"][f"{m}_corr"]+results_dict["none"][f"{m}_corr_err"], color="crimson", alpha=0.2)

        ax[ax_i].set_xticks(x_index)
        ax[ax_i].set_xticklabels(x_labels)
        ax[ax_i].set_ylabel(metric_titles[m],fontsize=44)
        #ax[ax_i].set_xlabel("Shuffle fraction",fontsize=44)
        ax[ax_i].tick_params(axis='both', which='major', labelsize=34,pad=10)
        ylow, yhigh = ax[ax_i].get_ylim()
        ax[ax_i].set_ylim(0.8*ylow,1.2*yhigh)
        ax_i += 1

    margins = {  #     vvv margin in inches
                "left"   : 0.14,
                "bottom" : 0.10,
                "right"  : 0.90,
                "top"    : 0.87,
                "wspace" : 0.35,
                "hspace": 0.4
                }

    fig.suptitle(dataset_title[dataset],fontsize=52,y=0.05)
    fig.legend(fontsize=35, loc='upper center',ncol=2)
    fig.subplots_adjust(**margins)
    fig.savefig(f"../plots/{dataset}_shuffle_metric_results{model_name}.pdf")


def main():
    parser = argparse.ArgumentParser(description='Plot performance metrics')
    parser.add_argument("--Data", required=True, choices=["CamCan","UKB"], help="Dataset")
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    args = parser.parse_args()

    plot(args.Data, args.Model)

if __name__ == '__main__':
    main()
