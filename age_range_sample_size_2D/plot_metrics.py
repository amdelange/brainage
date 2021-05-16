import sys, os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import xgboost as xgb
from reg_params import reg_params
from sklearn.model_selection import cross_val_predict
from dataset_names import dataset_names
from age_lims import age_lims, upper_age
import json
from downsample_fracs import downsample_fracs
import matplotlib.patheffects as PathEffects

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)

def plot(model):

    model_name = ""
    if(model!="XGB"):
        model_name = f"_{model}"

    metric_titles = {'r': "$r$",
                     'r2': "$R^2$",
                     'RMSE': "RMSE [years]",
                     'MAE': "MAE [years]",
                     }

    dataset = "UKB"
    file_name = dataset_names[dataset]

    for m in metric_titles:

        x_index = []
        x_label = []
        y_index = []
        y_label = []

        size_x = len(age_lims["lower"][dataset])
        size_y = len(downsample_fracs[dataset])

        vals = np.zeros((size_y, size_x))
        vals_corr = np.zeros((size_y, size_x))
        index = {}
        #Get values for each age range and sample size
        for i in range(0,size_x):
            x_index.append(size_x-i-1)
            x_label.append(f"{age_lims['lower'][dataset][i]}-{upper_age[dataset]}")
            for j in range(0,len(downsample_fracs[dataset])):

                with open(f"../output/{file_name}_lower_age_{age_lims['lower'][dataset][i]}_frac_{downsample_fracs[dataset][j]}_metrics{model_name}.json","r") as f:
                    results_dict = json.load(f)

                vals[j][i] = results_dict[m]
                vals_corr[j][i] = results_dict[f"{m}_corr"]

                if(i==0):
                    y_index.append(j)
                    y_label.append(f"{downsample_fracs[dataset][j]}")

        min_vals = np.min(vals)
        max_vals = np.max(vals)
        min_vals_corr = np.min(vals_corr)
        max_vals_corr = np.max(vals_corr)
        cmin = min_vals
        if(min_vals_corr < min_vals):
            cmin = min_vals_corr
        cmax = max_vals_corr
        if(max_vals > max_vals_corr):
            cmax = max_vals

        #Plot uncorrected values
        fig, ax = plt.subplots(figsize=(14,12))
        plt.set_cmap('GnBu_r')
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.set_xticks(x_index)
        ax.set_yticks(y_index)
        ax.set_xticklabels(x_label)
        ax.set_yticklabels(y_label)
        plt.xlabel("Age range [years]",fontsize=56)
        plt.ylabel("Sample fraction",fontsize=56)
        vals = np.flip(vals,1) #Flip the age range axis
        plt.imshow(vals,origin='lower')

        for i in range(0,size_y):
            for j in range(0,size_x):
                color = "k"
                if(vals[i][j] > 0.9*np.max(vals)):
                    color = "w"
                text = ax.text(j, i, "\\textbf{%.2f}" % vals[i][j],
                ha="center", va="center", color=color,fontsize=35)
                #text.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='w')])

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=40)
        #plt.clim(cmin,cmax)
        plt.title(metric_titles[m],fontsize=50,pad=20)
        plt.tight_layout()
        #plt.show()
        fig.savefig(f"../plots/additional_plots/{dataset}_2D_{m}_results{model_name}.pdf")



        #Plot corrected values
        fig, ax = plt.subplots(figsize=(14,12))
        plt.set_cmap('GnBu')
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.set_xticks(x_index)
        ax.set_yticks(y_index)
        ax.set_xticklabels(x_label)
        ax.set_yticklabels(y_label)
        plt.xlabel("Age range [years]",fontsize=56)
        plt.ylabel("Sample fraction",fontsize=56)
        vals_corr = np.flip(vals_corr,1) #Flip the age range axis
        plt.imshow(vals_corr,origin='lower')

        for i in range(0,size_y):
            for j in range(0,size_x):
                color = "k"
                if(vals_corr[i][j] > 0.94*np.max(vals_corr)):
                    color = "w"
                text = ax.text(j, i, "\\textbf{%.2f}" % vals_corr[i][j],
                ha="center", va="center", color=color,fontsize=35)
                #text.set_path_effects([PathEffects.withStroke(linewidth=0.25, foreground='w')])

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=40)
        #plt.clim(cmin,cmax)
        plt.title(metric_titles[m],fontsize=50,pad=20)
        plt.tight_layout()
        #plt.show()
        fig.savefig(f"../plots/additional_plots/{dataset}_2D_{m}_results_corr{model_name}.pdf")


def main():
    parser = argparse.ArgumentParser(description='Plot performance metrics')
    parser.add_argument("--Model", required=False, choices=["XGB","SVR"], help="Model", default="XGB")
    args = parser.parse_args()

    #Plot with and without correced values shown
    corr = ["N","Y"]
    for c in corr:
        plot(args.Model)

if __name__ == '__main__':
    main()
