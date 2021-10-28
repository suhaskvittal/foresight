"""
	author: Suhas Vittal
	date:	26 October 2021
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

import pandas as pd
import numpy as np

def load_classifier(data_file):
	df = pd.read_csv(data_file) 
	X = df[[
		'Layer Density, mean',\
		'Child Distance, mean'
	]].to_numpy()
	sabre_cnots = df['SABRE CNOTs']
	ips_cnots = df['MPATH_IPS CNOTs']
	bsp_cnots = df['MPATH_BSP CNOTs']
	y = np.zeros(sabre_cnots.shape[0])
	y[(sabre_cnots <= ips_cnots) & (sabre_cnots <= bsp_cnots)] = 0
	y[(ips_cnots < sabre_cnots) & (ips_cnots <= bsp_cnots)] = 1
	y[(bsp_cnots < sabre_cnots) & (bsp_cnots < ips_cnots)] = 2
	
	classifier = LogisticRegression(solver='newton-cg', multi_class='multinomial').fit(X, y)
	print(classifier.score(X, y))
	return classifier

