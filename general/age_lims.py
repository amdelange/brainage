
#Age limits to use when splitting the data into different ranges
age_lims = {}

#lower is the usual way i.e. 45-82, 50-82, 55-82 etc
age_lims["lower"] = {"CamCan": [60, 50, 40, 30, 20],
                    "UKB": [65, 60, 55, 50, 45]
                    }

#upper is the alternative way, 45-50, 45-55, 45-60 etc
age_lims["upper"] = {"CamCan": [30, 40, 50, 60, 87],
                    "UKB": [60, 65, 70, 75, 82]
                    }

#Highest age values in the samples
upper_age = {"CamCan": 87,
             "UKB": 82
            }

lower_age = {"CamCan": 18,
             "UKB": 45
            }
