"""
	author: Suhas Vittal
	date:	20 October 2021
"""

def _soln_hash_f(soln):
	h = 0
	PRIME = 5586537595543
	for (i, layer) in enumerate(soln):
		for (p0, p1) in layer:
			h += ((2**p0)*(3**p1)*(5**i)) % PRIME
	return h

def _path_hash_f(path):
	h = 0
	PRIME = 5586537595543
	for (p0, p1) in path:
		h += ((2**p0)*(3**p1)) % PRIME
	return h	

def _path_to_swap_collection(path):
	collection = []
	for (v1, v2) in path:
		collection.append([(v1, v2)])
	return collection

def _is_pow2(x):
	return (x & (x-1)) == 0
