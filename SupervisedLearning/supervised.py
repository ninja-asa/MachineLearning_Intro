# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:33:55 2017

@author: Sofia Assis

Dataset from kaggle
#,Name,Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary

"""
from sklearn import linear_model
import support as sp
from enum import Enum
import numpy as np

class Supervised:
	data = {}
	parameters = {}
	class Operation(Enum):
		LINEAR_REG = 0
		RIDGE_REG = 1
		
		
	def __init__(self,_data):
		self.data = _data
		
	def perform(self,_operation, _params):
		self.parameters = _params
		if (_operation == self.Operation.LINEAR_REG):
			return (self.linear_regression())
			
	def linear_regression(self):
		# Create feature matrix X
		if('features' in self.parameters.keys()):
			x_cols = self.parameters['features']
			print(x_cols)
		else:
			x_cols = [sp.DataStruct.ATK.value]
		if('output' in self.parameters.keys()):
			y_col = self.parameters['output']
		else:
			y_col = [sp.DataStruct.HP.value]
		X = np.zeros((len(self.data),len(x_cols)))
		counter = 0
		for pokemon in self.data:
			for feat in range(len(x_cols)):
				X[counter,feat] = pokemon[x_cols[feat]]
			counter = counter + 1
		
		y = np.zeros(len(self.data))
		counter = 0
		for pokemon in self.data:
			y[counter] = pokemon[y_col[0]]
			counter = counter + 1
		
		ls_solution = linear_model.LinearRegression(True,
									  normalize = True,
									  n_jobs = -1)
		
		ls_solution.fit(X,y)
		print (ls_solution.score(X, y, sample_weight=None))
		
		
		return ls_solution