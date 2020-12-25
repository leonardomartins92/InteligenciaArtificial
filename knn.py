# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:02:49 2020

@author: Leo
"""

import pandas as pd


base = pd.read_csv('microdados-matriculas.csv')
base = base.dropna()


##Dummys

base_dummy = pd.get_dummies(base, columns=['CategoriadeSituacao','Regiao','Sexo','RendaFamiliar','CorRaca','Turno'])
base_dummy = base_dummy.drop(columns=['Regiao_Região Centro-Oeste','CorRaca_Indígena','Sexo_F','RendaFamiliar_0<RFP<=0,5','CategoriadeSituacao_Evadidos','Turno_Matutino'])


classe = base_dummy.iloc[:, 4]
base_dummy = base_dummy.drop(columns=['CategoriadeSituacao_Concluintes'])
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
cm = confusion_matrix(y_train,result)

