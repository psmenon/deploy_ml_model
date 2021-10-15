import os
import yaml
import pytest
import pandas as pd
from src.evaluate import compute_model_metrics
from src.prepare import process_data

# data can be put into contest.py
data_columns = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'salary']

base_path = os.getcwd()
file = os.path.join(base_path,'config.yaml')

with open(file) as f:
    my_dict = yaml.safe_load(f)

cat_features = my_dict['model']['features']['categorical']
num_columns_after_process_data = 108

y_true = [1, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1]
model_scores = (0.3333333333333333, 0.3333333333333333, 0.3333333333333333)

@pytest.fixture(scope="session")
def data():

    df_path = os.path.join(base_path,f"data/prepared/{my_dict['data']['file_name']}")
    df = pd.read_csv(df_path)
    return df

def test_compute_model_metrics():
    
    result = compute_model_metrics(y_true,y_pred)
    assert result == model_scores
    
def test_data_columns(data):
    assert (data.columns == data_columns).all()

def test_process_data(data):
    num_columns = process_data(data,categorical_features=cat_features,label='salary',training=True)[0].shape[1]
    assert num_columns == num_columns_after_process_data