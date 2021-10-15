# Deploying a Machine Learning Model on Heroku with FastAPI

- Project **Deploying a Machine Learning Model on Heroku with FastAPI** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project we use data science to determine whether a person makes over 50K a year and then deploy the Machine Learning Model on Heroku with FastAPI.


## Files in Repository
```
* data
     * raw
       * census.csv.dvc
     * prepared
       * cleaned_census.csv.dvc
* api
     * main.py
* model
     * rfmodel_0.0.1.pkl.dvc
     * lb_0.0.1.pkl.dvc
     * encoder_0.0.1.pkl.dvc
* src
     * evaulate.py
     * prepare.py
     * train.py
* test
     * test_model.py
     * test_api.py
* metrics
     * rfmodel_0.0.1.results
* screenshots
    * continous_deployment.png
    * dvcdag.png
    * example.png
    * live_get.png
    * live_post.png
* run_pipeline.py
* config.yaml
* model_card.md
* setup.py
* README.md
```

## Usage

```python
Modify liveAPI_testing.py with your data and run

python liveAPI_testing.py
```

```bash
store data in a file called test.json (check liveAPI folder for format) and run

curl --request POST --header 'Content-Type: application/json' --data @test.json --url  https://udacity-ml-deploy.herokuapp.com/predict
```
