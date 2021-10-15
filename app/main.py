import os
import yaml
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel,Field

from src.prepare import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -r s3remote") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


# Loading in model from serialized .pkl file
base_path = os.getcwd()

file = os.path.join(base_path,'config.yaml')

with open(file) as f:
    config = yaml.safe_load(f)

model_name = config["model"]["model_name"]
version = config["model"]["version"]
cat_features = config['model']['features']['categorical']
    
pkl_filename = os.path.join(base_path,"model",f"{model_name}_{version}.pkl")
rf_model = joblib.load(pkl_filename)

encoder = joblib.load(os.path.join(base_path,"model",f"encoder_{version}.pkl"))
lb = joblib.load(os.path.join(base_path,"model",f"lb_{version}.pkl"))


class Census(BaseModel):
    age: int = Field(...,example=14)
    workclass: str = Field(...,example='Private')
    fnlgt: int = Field(...,example=45781)
    education: str = Field(...,example='Masters')
    education_num: int = Field(alias='education-num',example=14)
    marital_status: str = Field(alias="marital-status",example='Never-married')
    occupation: str = Field(...,example='Prof-Speciality')
    relationship: str = Field(...,example='Not-in-family')
    race: str = Field(...,example='White')
    sex: str = Field(...,example='Male')
    capital_gain: str = Field(alias="capital-gain",example=14567)
    capital_loss: str = Field(alias="capital-loss",example=0)
    hours_per_week: str = Field(alias="hours-per-week",example=40)
    native_country: str = Field(alias="native-country",example='India')
 

@app.get("/")
async def welcome():
    return {"message": "Welcome to Udacity Model Deployment Project"}


@app.post('/predict')
async def predict(census: Census):
 
    # Converting input data into Pandas DataFrame
    input_df = pd.DataFrame([census.dict(by_alias=True)])
    
    input_test, _, _, _ = process_data(
        input_df, categorical_features=cat_features, label=None, training=False,encoder=encoder,lb=lb
    )
 
    # Getting the prediction from the Random Forest model
    pred = rf_model.predict(input_test).tolist()[0]
 
    return pred
