# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:38:03 2016

@author: samuele

Porting to json
"""
import sys
import json
import os

"""this allow to read another json file and "copy" the same setting"""

def retreiveParamsFromFile(jsonPath):
    with open(jsonPath) as data_file:    
        data = json.load(data_file)
        
    for key, val in data.items():
        if key=="supervised_algorithm":
            for k, v in val.items():
                if key=="verbose":
                    globals()["supervised_verbose"] = v
                else:
                    globals()[k] = v
        elif key=="rl_algorithm":
            for k, v in val.items():
                if key=="verbose":
                    globals()["unsupervised_verbose"] = v
                else:
                    globals()[k] = v
        else:
            for k, v in val.items():
                globals()["unsupervised_verbose"] = v

    
    return data
    
def retreiveParams():
    global folder,test,case,code, number, library
    commands = sys.argv[1:]
        
    for arg in commands:
        s = arg.split("=")    
        name = s[0]
        value = s[1]
        if name=="copy_from":
            retreiveParamsFromFile(value)
        else:
            try:
                if "." in value:
                    globals()[name] = float(value)
                else:
                    globals()[name] = int(value)
            except:
                globals()[name] = str(value)
            
json_path="experiment.json"
name="exp1"
load_path = ""
save_path = ""
n_dataset = 20
n_experiments = 1
model_name = "mlp"
n_hidden_neurons = 10
n_layers = 2
optimizer= "rmsprop"
activation="relu"
mdp_name="MountainCar"
n_epoch=300
batch_size=100
validation_split=0.1
supervised_verbosity=0
unsupervised_verbosity=0
n_iterations=20
gamma=0.98
horizon = 50000
scaled = 1

retreiveParams()

json_file = {
    "experiment": {
        "name":name,
        "date":"..."
    },
    "experiment_setting":{
        "load_path":load_path ,
        "save_path":save_path,
        "n_datasets":n_dataset,
        "n_experiments":n_experiments
    },
    "model":{
        "model_name":model_name,
        "n_hidden_neurons":n_hidden_neurons,
        "n_layers":n_layers,
        "optimizer":optimizer,
        "activation":activation
    },
    "mdp":{
        "mdp_name":mdp_name    
    },
    "supervised_algorithm":{
        "n_epochs":n_epoch,
        "batch_size":batch_size,
        "validation_split":validation_split,
        "verbosity":supervised_verbosity
    },
    "rl_algorithm":{
        "algorithm":"FQI",
        "n_iterations":n_iterations,
        "gamma":gamma,      
        "verbosity":unsupervised_verbosity,
        "horizon":horizon,
        "scaled":scaled
    }
}

with open(json_path, 'w') as fp:
    json.dump(json_file, fp)

cmd = "python experiment.py " + json_path
os.system(cmd)
