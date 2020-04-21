from .data_preparation import data_transform
from sklearn.ensemble import RandomForestClassifier


def prediction_model(x):
    x = x.copy()
    Xtrain, ytrain, Xtest = data_transform(x)
    
    mdl = RandomForestClassifier(max_depth=269,
                                 random_state=0,
                                 n_estimators=5134,
                                 max_features='sqrt', 
                                 min_samples_split=15,
                                 min_samples_leaf=1,
                                 bootstrap=True, n_jobs=4)
    
    mdl.fit(Xtrain, ytrain)
    prediction = mdl.predict(Xtest)
    if prediction[-1] == 0:
        return "Not survived"
    elif prediction[-1] == 1:
        return "Survived"
    else:
        prediction = "Eror"
    return prediction