"""
	author: Suhas Vittal
	date: 	15 October 2021
"""

import pandas as pd

def build_dict(csv_file):
	df = pd.read_csv(csv_file)		

	print(df)
