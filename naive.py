# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:29:06 2019

@author: leohe
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB

labelencoder = LabelEncoder()

base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:,4].values

for i in range(len(previsores[0])):
    previsores[:,i] = labelencoder.fit_transform(previsores[:,i]) #fit_transform transforma para variavel discreta

GNB = GaussianNB() #classificador
GNB.fit(previsores, classe)

resultado = GNB.predict([[0,0,1,2], [3,0,0,0]])

