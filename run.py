
from train_sklearn import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, PassiveAggressiveRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb


def writeResultModel(modelname, perf, params, other):

    f = open('stack_result.txt','a')
    f.write(modelname+ '\n')
    f.write(other+ '\n')
    f.write(str(perf)+ 'RMSLE \n')
    f.write(str(params)+ '\n')
    f.write('\n')
    f.close()


# #
# model = LinearRegression()
# params = {}
# modelname = "LinearRegression_base"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, datasetRead="base")
# other = "nbags=1"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)

#
model = LinearRegression()
params = {}
modelname = "LinearRegression_add"
perf = trainModel(model, params=params, nbags=1, modelname = modelname, datasetRead="add")
other = "nbags=1"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)


#
model = xgb.XGBRegressor()
modelname = "XGBRegressor_base"
params = {"n_estimators": 1500, "nthread":-1, "colsample_bytree":0.9, "subsample":0.9, "reg_alpha":5, "reg_lambda":3}
perf = trainModel(model, params=params, nbags=1, modelname = modelname, datasetRead="base")
other = "nbags=1"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)

#
model = xgb.XGBRegressor()
modelname = "XGBRegressor_add"
params = {"n_estimators": 1500, "nthread":-1, "colsample_bytree":0.9, "subsample":0.9, "reg_alpha":5, "reg_lambda":3}
perf = trainModel(model, params=params, nbags=1, modelname = modelname, datasetRead="add")
other = "nbags=1"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)

#
model = xgb.XGBRegressor()
modelname = "XGBRegressor_add_bag"
params = {"n_estimators": 1500, "nthread":-1, "colsample_bytree":0.9, "subsample":0.9, "reg_alpha":5, "reg_lambda":3}
perf = trainModel(model, params=params, nbags=5, modelname = modelname, datasetRead="add")
other = "nbags=5"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)

