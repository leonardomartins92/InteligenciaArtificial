# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:02:49 2020

@author: Leo
"""

import pandas as pd

base = pd.read_csv('alunos_gret.csv')
base = base.fillna(method='ffill')
##Dummys

base_dummy = pd.get_dummies(base, columns=['CategoriadeSituacao','Sexo','RendaFamiliar','CorRaca','FaixaEtaria','Turno','Regiao'])
classe = base_dummy.iloc[:, 4]
base_dummy = base_dummy.drop(columns=['CategoriadeSituacao_Evadidos'])
variavel = base_dummy.iloc[:, 0:]

#Normalização
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
variavel_standard = scaler.fit_transform(variavel)

#Separando em grupos teste e treinamento
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(variavel, classe, test_size=0.25,random_state=0)


#KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
result = knn.predict(X_train)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_train,result)

from sklearn.model_selection import GridSearchCV
k_list = list(range(1,31))
parametros = dict(n_neighbors=k_list)
grid = GridSearchCV(knn, variavel, cv=5, scoring='accuracy')
grid.fit(variavel, classe)
