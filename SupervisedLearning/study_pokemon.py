# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:17:48 2017

@author: Sofia Assis
"""

import numpy as np
import csv
import supervised
import support as sp
if __name__=="__main__":
	with open('pokemon.csv', 'rb') as f:
	    reader = csv.reader(f)
	    pokemon_list = list(reader)
	
	pokemon_list = pokemon_list[1:]
	A = supervised.Supervised(pokemon_list)
	A.data = pokemon_list
#	print('oi')
	d  = {'features':[sp.DataStruct.ATK.value, sp.DataStruct.DEF.value], 'output': [sp.DataStruct.HP.value]}
	aaaa = A.perform(supervised.Supervised.Operation.LINEAR_REG,d)
#	print()