from fastapi import FastAPI
from pydantic import BaseModel,Field

app = FastAPI()

class Census(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: str = Field(alias="capital-gain")
    capital_loss: str = Field(alias="capital-loss")
    hours_per_week: str = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
 

@app.get("/")
async def welcome():
    return {"message": "Welcome to Udacity Model Deployment Project"}


@app.post('/predict')
async def predict(census: Census):
 
    # Converting input data into Pandas DataFrame
    input_df = pd.DataFrame([census.dict()])
 
    # Getting the prediction from the Random Forest model
    pred = rf_model.predict(input_df)[0]
 
    return pred
