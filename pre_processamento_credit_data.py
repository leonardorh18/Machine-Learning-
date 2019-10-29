# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd

base =pd.read_csv('credit-data.csv')
base.describe()
base.loc[base['age'] < 0] #localiza idades inconsistentes, ou seja, idades negativas
#deleta columa idade  = base.drop('age',1,inplace = True) 

#base.drop(base[base.age < 0].index, inplace= True) #deleta idades negativas

#preencher idades negativas com a media

# base.mean()  # base.['age'].mean()

base['age'][base.age > 0].mean()   #media só das idades maiores que 0, sem o [age] tira de todas os dados
base.loc[base.age < 0, 'age'] = 40.92  #atualiza as idades negativas com o vaor da media das idades

#localizar valores null
# pd.isnull(base['age']) mostra tudo com True e False, mais complicado
base.loc[pd.isnull(base['age'])] 

previsores = base.iloc[:,1:4].values #pega os valores das colunas 1-3 pq 4 não entra, começa no 0
classe = base.iloc[:, 4].values #[:, 4] o ' : ' significa todas as linhas

from sklearn.preprocessing import Imputer, StandardScaler
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

# scaler

scaler =  StandardScaler()
previsores = scaler.fit_transform(previsores) #padronização dos cados = colocar em uma mesma escala

























