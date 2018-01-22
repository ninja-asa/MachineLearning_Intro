# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:17:48 2017

@author: Sofia Assis
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import supervised
import support as sp
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    #Load Pokedex
    with open('pokemon.csv', 'rt') as f:
        reader = csv.reader(f)
        pokemon_list = list(reader)
    #Discard description line
    pokemon_list = pokemon_list[1:]

    #todo: train and test split

    #Let us do some data exploration
    # Starting with Principal Component Analysis. For such purpose let us build
    # an array with every numerical feature
    numerical_indices = np.array([sp.DataStruct.ATK.value,
                                  sp.DataStruct.DEF.value,
                                  sp.DataStruct.HP.value,
                                  sp.DataStruct.SP_ATK.value,
                                  sp.DataStruct.SP_DEF.value,
                                  sp.DataStruct.SPEED.value],dtype=np.int)
    numerical_data = np.array([[pok[i] for i in numerical_indices] for pok in pokemon_list])
    
    model_lsm = supervised.Supervised(pokemon_list)
    model_lsm.data = pokemon_list
    #todo: perform linear regression between all combinations and select top five regressions
    data_input = {'features':[sp.DataStruct.ATK.value], #, sp.DataStruct.DEF.value
         'output': [sp.DataStruct.HP.value]}
    ls_error = model_lsm.perform(supervised.Supervised.Operation.LINEAR_REG, data_input)
    y_pred = model_lsm.predict()
    print('Coefficients: \n', model_lsm.coef)
    print("Mean squared error: %.2f"
        % mean_squared_error(model_lsm.train_y, y_pred))
    print('Variance score: %.2f' % r2_score(model_lsm.train_y, y_pred))

    # Plot outputs
    plt.scatter(model_lsm.train_X, model_lsm.train_y,  color='black')
    plt.plot(model_lsm.train_X, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    
    