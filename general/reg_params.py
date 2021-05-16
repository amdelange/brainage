
#Parameters for the regression model
reg_params = {}

reg_params["UKB"] = {'n_estimators': 180,
              'max_depth': 3,
              'learning_rate': 0.1,
              'n_cv': 10,
              'n_jobs': 4
              }

reg_params["CamCan"] = {'n_estimators': 180,
              'max_depth': 3,
              'learning_rate': 0.1,
              'n_cv': 10,
              'n_jobs': 4
              }
