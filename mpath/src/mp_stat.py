"""
    author: Suhas Vittal
    date:   26 October 2021
"""

from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from mp_util import _compute_per_layer_density_2q,\
                    _compute_child_distance_2q,\
                    _compute_size_depth_ratio_2q,\
                    _compute_in_layer_qubit_distance_2q

import pandas as pd
import numpy as np

def load_classifier(data_file):
    df = pd.read_csv(data_file) 
    X = df[[
        'Layer Density, mean',\
        'Layer Density, std.',\
        'Child Distance, mean',\
        'Child Distance, std.',\
        'In Layer Distance, mean',\
        'In Layer Distance, std.'
    ]].to_numpy()
    sabre_cnots = df['SABRE CNOTs']
    ips_cnots = df['MPATH_IPS CNOTs']
    # Declare response
    y = np.zeros(sabre_cnots.shape[0])
    y[sabre_cnots > ips_cnots] = 1
    
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), SVC(
        max_iter=100000
    ))
    clf = clf.fit(X, y)
    print('classifier score: ', clf.score(X, y))
    return clf

def load_regressor(data_file):  
    df = pd.read_csv(data_file) 
    X = df[[
        'Layer Density, mean',\
        'Layer Density, std.',\
        'Child Distance, mean',\
        'Child Distance, std.',\
        'In Layer Distance, mean',\
        'In Layer Distance, std.'
    ]].to_numpy()
    sabre_cnots = df['SABRE CNOTs']
    ips_cnots = df['MPATH_IPS CNOTs']
    # Declare response
    y = (ips_cnots - sabre_cnots).to_numpy()
    y = y / np.max(np.abs(y))
    
    reg = make_pipeline(StandardScaler(with_mean=True, with_std=True), SGDRegressor(
        learning_rate='optimal',
        max_iter=100000
    ))
#    reg = make_pipeline(StandardScaler(with_mean=True, with_std=True), SVR(
#       max_iter=100000
#    ))
    reg = reg.fit(X, y)
    print('regressor score: ', reg.score(X, y))

    return reg

def get_independent_variable(primary_layer_view):
    dens_mean, dens_std = _compute_per_layer_density_2q(primary_layer_view, weighted=True)
    cdist_mean, cdist_std = _compute_child_distance_2q(primary_layer_view, weighted=True)
    idist_mean, idist_std = _compute_in_layer_qubit_distance_2q(primary_layer_view, weighted=True) 
    X = [[
        dens_mean,
        dens_std,
        cdist_mean,
        cdist_std,
        idist_mean,
        idist_std
    ]]
    return np.array(X)
