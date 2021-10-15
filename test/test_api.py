from fastapi.testclient import TestClient

import pytest
import json
from app.main import app

client = TestClient(app)

test_data = [({
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 2174,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
},0),
    ({
  "age": 31,
  "workclass": "Private",
  "fnlgt": 45781,
  "education": "Masters",
  "education-num": 14,
  "marital-status": "Never-married",
  "occupation": "Prof-speciality",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Female",
  "capital-gain": 14084,
  "capital-loss": 0,
  "hours-per-week": 50,
  "native-country": "United-States"
},1)]

    


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Udacity Model Deployment Project"}
    
@pytest.mark.parametrize("data, result",test_data)    
def test_predict(data,result):
    r = client.post("/predict",json=data)
    assert r.status_code == 200
    assert json.loads(r.content) == result
    
    
def test_malformed_predict():
    r = client.post("/predict",json={})
    assert r.status_code != 200
    
    

