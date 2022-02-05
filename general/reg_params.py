#Parameters for the regression models
reg_params = {}

reg_params["UKB"] = {'n_estimators': 150,
              'max_depth': 5,
              'learning_rate': 0.1,
              'n_cv': 10,
              'n_jobs': 4,
              'C': 1.
              }

reg_params["CamCan"] = reg_params["UKB"]
