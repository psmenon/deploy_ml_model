# Model Card


## Model Details
Pranav Sivadas Menon created the model. It is a random forest model using the default hyperparameters in scikit-learn 1.0


## Intended Use
This model should be used to predict whether income exceeds $50K/yr based on census data. The users are people working in the Census Bureau.


## Training Data
The data was obtained from the UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/census+income. The data has 32562 instances and 14 attributes. The target variable is salary.To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.



## Evaluation Data
The data was split such that 80% of data was used in the training process and the remaining 20 percent was used in testing. No stratification was done


## Metrics
Navigate to the metrics folder to get the precision,recall and fbeta score on the various slices of the data


## Ethical Considerations
Bias may be present in the model. The data may have been collected from several countries,however the proportion is not the same. Additional testing may be required based on the region the model is used.


## Caveats and Recommendations
Only random forest model was tried on the data.Other models may work better
No hyperparameter tuning was done and should be performed to get a better model
Use Aequitas to check if bias is present in the model

