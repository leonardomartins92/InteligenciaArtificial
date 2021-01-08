# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:49:24 2020

@author: Leo
"""

import pandas as pd


df = pd.read_csv('dataset/CSV/microdados-matriculas.csv')
df = df.dropna()


##SETTING DUMMYS 

df_dummy = pd.get_dummies(df, columns=['CategoriadeSituacao','Regiao','Sexo','RendaFamiliar','CorRaca','Turno'])
df_dummy = df_dummy.drop(columns=['Regiao_Região Centro-Oeste','CorRaca_Indígena','Sexo_F','RendaFamiliar_0<RFP<=0,5','CategoriadeSituacao_Evadidos','Turno_Matutino'])


y = df_dummy.iloc[:, 4]
df_dummy = df_dummy.drop(columns=['CategoriadeSituacao_Concluintes'])
X = df_dummy.iloc[:, 0:]
                    
#SLIPTTING TEST AND TRAINING GROUP 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# NORMALIZATION
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#TUNING

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(11, kernel_initializer='glorot_uniform', activation = 'relu', input_dim = 22)) 
    classifier.add(Dense(11, kernel_initializer='glorot_uniform', activation = 'relu')) 
    classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation = 'sigmoid')) 
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

params = {'batch_size':[25, 32, 50],'epochs':[50, 100, 500], 'optimizer':['adam', 'rmsprop'] }

grid_search = GridSearchCV(estimator = classifier, param_grid = params, 
                           scoring='accuracy', cv=10)

grid_search = grid_search.fit(x_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
