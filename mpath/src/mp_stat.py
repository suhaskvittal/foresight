"""
	author: Suhas Vittal
	date:	26 October 2021
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
        'Size-Depth Ratio',\
        'In Layer Distance, mean',\
        'In Layer Distance, std.',\
	]].to_numpy()
	sabre_cnots = df['SABRE CNOTs']
	ips_cnots = df['MPATH_IPS CNOTs']
	# Declare response
	y = np.zeros(sabre_cnots.shape[0])
	y[sabre_cnots > ips_cnots] = 1
	
	clf = make_pipeline(StandardScaler(), SVC())
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
        'Size-Depth Ratio',\
        'In Layer Distance, mean',\
        'In Layer Distance, std.',\
	]].to_numpy()
	sabre_cnots = df['SABRE CNOTs']
	ips_cnots = df['MPATH_IPS CNOTs']
	# Declare response
	y = np.zeros(sabre_cnots.shape[0])
	y = ips_cnots - sabre_cnots
	
	reg = make_pipeline(StandardScaler(), SGDRegressor(
        alpha=0.01,
        learning_rate='optimal',
        early_stopping=True,
        validation_fraction=0.1
    ))
	reg = reg.fit(X, y)
	print('regressor score: ', reg.score(X, y))

	return reg

def get_independent_variable(primary_layer_view):
    dens_mean, dens_std = _compute_layer_density_2q(primary_layer_view)
    cdist_mean, cdist_std = _compute_child_distance_2q(primary_layer_view)
    size_depth_ratio = _compute_size_depth_ratio_2q(primary_layer_view)
    idist_mean, idist_std = _compute_in_layer_qubit_distance_2q(primary_layer_view) 
    X = [[
        dens_mean,
        dens_std,
        cdist_mean,
        cdist_std,
        size_depth_ratio,
        idist_mean,
        idist_std
    ]]
    return np.array(X)
