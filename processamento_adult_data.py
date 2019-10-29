# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:18:26 2019

@author: leohe
"""

import pandas as pd
base = pd.read_csv('census.csv')
previsores = base.iloc[:,0:14].values #13 elementos mas 14 pra pegar o 13
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

LEncoder = LabelEncoder()
#labels = LEncoder_previsores.fit_transform(previsores[:,1]) #transforma a variavel ordinal em classifcação por numero - discreta

previsores[:, 1] = LEncoder.fit_transform(previsores[:,1])
previsores[:, 3] = LEncoder.fit_transform(previsores[:,3])
previsores[:, 5] = LEncoder.fit_transform(previsores[:,5])
previsores[:, 6] = LEncoder.fit_transform(previsores[:,6])
previsores[:, 7] = LEncoder.fit_transform(previsores[:,7])
previsores[:, 8] = LEncoder.fit_transform(previsores[:,8])
previsores[:, 9] = LEncoder.fit_transform(previsores[:,9])
previsores[:, 13] = LEncoder.fit_transform(previsores[:,13])

onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

classe = LEncoder.fit_transform(classe)

from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
previsores = scaler.fit_transform(previsores)























