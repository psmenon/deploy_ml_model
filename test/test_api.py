from fastapi.testclient import TestClient

import json
from api.main import app

client = TestClient(app)

test_data = {
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
}


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    
    
def test_predict():
    r = client.post("/predict",json=test_data)
    assert r.status_code == 200
    assert json.loads(r.content) == 0
    
    
def test_malformed_predict():
    r = client.post("/predict",json={})
    assert r.status_code != 200
    
    

