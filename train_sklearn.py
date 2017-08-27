import read_data
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, f_regression
from sklearn.cross_validation import StratifiedKFold

def list_to_percentiles(numbers):
    pairs = zip(numbers, range(len(numbers)))
    pairs.sort(key=lambda p: p[0])
    result = [0 for i in range(len(numbers))]
    for rank in xrange(len(numbers)):
        original_index = pairs[rank][1]
        result[original_index] = int( rank * 100.0 / (len(numbers)-1))
    return result

def trainModel(model, datasetRead = "base", modelname= "", nbags = 1,
               params = {}):


    if datasetRead == "base":
        xtrain, xtest, id_train, id_test, y = read_data.read_data_base()
    elif datasetRead == "add":
        xtrain, xtest, id_train, id_test, y = read_data.read_data_add_features()


    xtrain = np.array(xtrain)
    xtest = np.array(xtest)


    ## cv-folds
    nfolds = 5
    lossl = list_to_percentiles(np.log(y+1).ravel())
    folds = StratifiedKFold(lossl, n_folds=5, shuffle = True, random_state = 20)

    ## train models
    i = 0

    pred_oob = np.zeros(xtrain.shape[0])
    pred_test = np.zeros(xtest.shape[0])

    for (inTr, inTe) in folds:
        xtr = xtrain[inTr]
        xte = xtrain[inTe]

        ytr = np.log(y[inTr]+1).ravel()
        yte = np.log(y[inTe]+1).ravel()


        pred = np.zeros(xte.shape[0])
        for j in range(nbags):

            #params['random_state'] = 1337+j

            model.set_params(**params)
            model.fit(xtr, ytr)

            pred += np.exp(model.predict(xte) )-1
            pred_test += np.exp(model.predict(xtest))-1

        pred /= nbags
        pred_oob[inTe] = pred
        i += 1
        #print('Fold ', i, '- MAE:', score)
    pred_test /= (nfolds*nbags)

    #print('Total - MAE:', mean_absolute_error(y, pred_oob))
    ## train predictions
    df = pd.DataFrame({'id': id_train, 'trip_duration': pred_oob})


    perf = mean_squared_error(np.log(y+1), np.log(np.clip(pred_oob, 0, None) +1))

    df.to_csv('preds/preds_oob_'+modelname+str(perf)+'.csv', index = False)

    ## test predictions



    df = pd.DataFrame({'id': id_test, 'trip_duration': np.clip(pred_test, 0, None)})
    df.to_csv('preds/submission_'+modelname+str(perf)+'.csv', index = False)

    return perf