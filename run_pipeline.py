import hydra
from omegaconf import DictConfig, OmegaConf

import os
from joblib import dump
import pandas as pd
import numpy as np

from src.prepare import process_data
from src.train import train_model
from src.evaluate import inference,compute_model_metrics,slice_census

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import logging

logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@hydra.main(config_name='config')
def main(config: DictConfig):
    
    root_path = hydra.utils.get_original_cwd()
    
    try:
        data = pd.read_csv(os.path.join(f"{root_path}/data/prepared",config["data"]["file_name"]))
    except FileNotFoundError as e:
        logging.info("Filename specified does not Exist")
        return

    train, test = train_test_split(data, test_size=0.20)

    cat_features = config["model"]["features"]["categorical"]

    # Train model
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    randomforest = RandomForestClassifier()
    model = train_model(randomforest,X_train,y_train)
    
    model_name = config["model"]["model_name"]
    version = config["model"]["version"]
   
    with open(os.path.join(root_path,"metrics",f"{model_name}_{version}.results"), "a") as file:
        for feature in cat_features:
            class_dict = slice_census(model,encoder,lb,test,feature,cat_features)
            file.write(f"{feature}:: {class_dict}")
            file.write("\n")
            file.write("\n")
    
   
    dump(model,os.path.join(root_path,"model",f"{model_name}_{version}.pkl"))
    dump(encoder,os.path.join(root_path,"model",f"encoder_{version}.pkl"))
    dump(lb,os.path.join(root_path,"model",f"lb_{version}.pkl"))
         
         
    logging.info("pkls generated")
    
if __name__  == "__main__":
    main()