import sys, os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import xgboost as xgb
from reg_params import reg_params
from sklearn.model_selection import cross_val_predict
from dataset_names import dataset_names
from age_lims import age_lims, upper_age, lower_age
import json

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

metric_titles = {'r': "$r$",
                 'r2': "$R^2$",
                 'RMSE': "RMSE [years]",
                 'MAE': "MAE [years]",
                 #'MedianAE': "Median AE",
                 #"WMAE" : "$_{W}$MAE",
                 #'RSE': "RSE",
                 #'RAE': "RAE"
                 }

dataset_title = {'CamCan': 'Cam-CAN',
              'UKB': 'UKB'
                 }


def plot(dataset,model,agerange,corr):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    age_ranges = []#["all"]
    for a in age_lims[agerange][dataset]:
        age_ranges.append(str(a))

    x_index = []
    x_labels = []

    for i in range(0, len(age_lims[agerange][dataset])):
        x_index.append(i)
        if(agerange=="lower"):
            x_labels.append("%s-%s" % (int(age_lims[agerange][dataset][i]),int(upper_age[dataset])))
        else:
            x_labels.append("%s-%s" % (int(lower_age[dataset]),int(age_lims[agerange][dataset][i])))
    x_labels.reverse()

    results_dict = {}
    for a in age_ranges:

        if(a=="all"):
            suff = ""
        else:
            suff = f"_{agerange}_age_{a}"

        file_name = f"../output/{dataset_names[dataset]}{suff}_metrics{model_name}_fixed_test_range.json"

        with open(file_name,"r") as f:
            results_dict[a] = json.load(f)

    #Loop over metrics
    fig, ax = plt.subplots(2,2, figsize=(18,16))
    ax = ax.ravel()

    ax_i = 0

    for m in metric_titles:

        y_vals = []
        y_errs = []
        y_vals_corr = []
        y_errs_corr = []


        for a in age_lims[agerange][dataset]:

            val = results_dict[str(a)][m]
            y_vals.append(val)
            err = results_dict[str(a)][f"{m}_err"]
            y_errs.append(err)

            val_corr = results_dict[str(a)][f"{m}_corr"]
            y_vals_corr.append(val_corr)
            err_corr = results_dict[str(a)][f"{m}_corr_err"]
            y_errs_corr.append(err_corr)

        lab = ""
        label = ""
        corr_label = ""
        if(ax_i==0):
            lab = "models (N = %.0f)" % results_dict[str(age_lims[agerange][dataset][-1])]["n"]
            label = "Age range "+lab
            corr_label = "Corr. age range "+lab
        #Reverse values
        y_vals.reverse()
        y_vals_corr.reverse()

        #Plot uncorrected values
        ax[ax_i].errorbar(x_index,y_vals,yerr=y_errs,fmt="o",markersize=6,capsize=8,color="k",label=label)
        #Plot corrected values
        if(corr=="Y"):
            ax[ax_i].errorbar(x_index,y_vals_corr,yerr=y_errs_corr,fmt="o",markersize=6,capsize=8,color="crimson",label=corr_label)

        #Plot result of full model to compare
        line_lab = ""
        line_label = ""
        corr_line_label = ""
        if(ax_i==0):
            line_lab = "model (N = %.0f)" % results_dict[str(age_lims[agerange][dataset][0])]["n"]
            line_label = "Full "+line_lab
            corr_line_label = "Corr. full "+line_lab

        ax[ax_i].set_xticks(x_index)
        ax[ax_i].set_xticklabels(x_labels)
        ax[ax_i].set_ylabel(metric_titles[m],fontsize=44)
        ax[ax_i].tick_params(axis='both', which='major', labelsize=34,pad=10)
        ax[ax_i].grid(which='major', axis='y', linestyle='-')
        ylow, yhigh = ax[ax_i].get_ylim()
        ax_i += 1

    margins = {  #     vvv margin in inches
                "left"   : 0.14,
                "bottom" : 0.14,
                "right"  : 0.88,
                "top"    : 0.85,
                "wspace" : 0.35,
                "hspace": 0.4
                }

    fig.suptitle(dataset_title[dataset],fontsize=52,y=0.06)
    fig.legend(fontsize=38, loc='upper center',ncol=1)
    fig.subplots_adjust(**margins)
    fig.savefig(f"../plots/additional_plots/{dataset}_metric_results_corr_{corr}{model_name}_age_{agerange}_fixed_test_range.pdf")


def main():
    parser = argparse.ArgumentParser(description='Plot performance metrics')
    parser.add_argument("--Data", required=True, choices=["CamCan","UKB"], help="Dataset")
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    parser.add_argument("--AgeRange", required=True, choices=["lower","upper"], help="Whether to move lower or upper age")
    args = parser.parse_args()

    #Plot with and without correced values shown
    corr = ["N","Y"]
    for c in corr:
        plot(args.Data,args.Model,args.AgeRange,c)

if __name__ == '__main__':
    main()
