import requests
import json

data = {
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
}

response = requests.post(' https://udacity-ml-deploy.herokuapp.com/predict',data=json.dumps(data))

if response.status_code == 200:
    if json.loads(response.content) == 1:
        print('Salary >50k')
    else:
        print('salary <=50k')  
else:
    print(f'Status code {response.status_code} raised')