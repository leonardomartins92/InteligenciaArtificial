
import pandas as pd


base = pd.read_csv('microdados-matriculas.csv')
base = base.dropna()


##Dummys

base_dummy = pd.get_dummies(base, columns=['CategoriadeSituacao','Regiao','Sexo','RendaFamiliar','CorRaca','Turno'])
base_dummy = base_dummy.drop(columns=['Regiao_Região Centro-Oeste','CorRaca_Indígena','Sexo_F','RendaFamiliar_0<RFP<=0,5','CategoriadeSituacao_Evadidos','Turno_Matutino'])


classe = base_dummy.iloc[:, 4]
base_dummy = base_dummy.drop(columns=['CategoriadeSituacao_Concluintes'])
variavel = base_dummy.iloc[:, 0:]
                    
#Separando em grupo de teste e de treinamento
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(variavel, classe, test_size=0.2,random_state=0)


# Arrumando Escala
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

#INPUT LAYER
#primeiro valor = (variaveis+classe)/2 

classifier.add(Dense(11, kernel_initializer='glorot_uniform', activation = 'relu', input_dim = 22)) 
classifier.add(Dropout(0.05))

classifier.add(Dense(11, kernel_initializer='glorot_uniform', activation = 'relu')) 
classifier.add(Dropout(0.05))

#OUTPUT LAYER
#como a saida tem 1 variavel categoria, o primeiro numero é 1 e a activação é sigmoid.
# Se tivesse mais de 1 a ativação seria softmax e o valor inicial acompanharia a qtd
classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation = 'sigmoid')) 

#COMPILAR 
classifier.compile(optimizer ='rmsprop', loss='binary_crossentropy',metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 50, epochs = 50)
#TESTE
y_pred = classifier.predict(x_test)
y_pred2 = (y_pred > 0.50)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)


#K-FOLD CROSS VALIDATION

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(11, kernel_initializer='glorot_uniform', activation = 'relu', input_dim = 22)) 
    classifier.add(Dense(11, kernel_initializer='glorot_uniform', activation = 'relu')) 
    classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation = 'sigmoid')) 
    classifier.compile(optimizer ='adam', loss='binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 50, epochs = 50)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = -1)    
mean = accuracies.mean()
variance = accuracies.std()


