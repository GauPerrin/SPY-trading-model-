#Py file to have the function ready for any other model 

import pandas as pd

def adjustedMetric(data, model, model_k, yname):
    data["yhat"] = model.predict(data)
    SST = ((data[yname] - data[yname].mean()) ** 2).sum()
    SSR = ((data["yhat"] - data[yname].mean()) ** 2).sum()
    SSE = ((data[yname] - data["yhat"]) ** 2).sum()
    r2 = SSR / SST
    adjustR2 = 1 - (1 - r2) * (data.shape[0] - 1) / (data.shape[0] - model_k - 1)
    RMSE = (SSE / (data.shape[0] - model_k - 1)) ** 0.5
    return adjustR2, RMSE
# Root Mean Squared Error and Adjusted R2
# model_k is the number of predictors
# ynam is the column name of our response variable
# model is the model name here lm


def assessTable(test, train, model, model_k, yname):
    r2test, RMSEtest = adjustedMetric(test, model, model_k, yname)
    r2train, RMSEtrain = adjustedMetric(train, model, model_k, yname)
    assessment = pd.DataFrame(index=["R2", "RMSE"], columns=["train", "test"])
    assessment["train"] = [r2train, RMSEtrain]
    assessment["test"] = [r2test, RMSEtest]
    return assessment
