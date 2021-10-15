
def train_model(model,X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    model: 
         machine learning model
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model.fit(X_train,y_train)
    return model