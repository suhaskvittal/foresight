"""
	author: Suhas Vittal
	date:	26 October 2021
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

import pandas as pd
import numpy as np

def load_classifier(data_file):
	df = pd.read_csv(data_file) 
	X = df[[
		'Layer Density, mean',\
		'Layer Density, std.',\
		'Child Distance, mean',\
		'Child Distance, std.'
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
		'Child Distance, mean'
	]].to_numpy()
	sabre_cnots = df['SABRE CNOTs']
	ips_cnots = df['MPATH_IPS CNOTs']
	# Declare response
	y = np.zeros(sabre_cnots.shape[0])
	y = sabre_cnots - ips_cnots
	
	reg = make_pipeline(StandardScaler(), SVR(
		gamma=1,
		C=1,
		epsilon=0.2
	))
	reg = reg.fit(X, y)
	print(reg.score(X, y))
	exit()
	return reg

