from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABC, abstractmethod, abstractstaticmethod
import numpy as np
from sklearn.preprocessing import label_binarize

from src.util import sign

class Perceptron(BaseEstimator, ClassifierMixin):
    '''
    Deinfinição da rede Perceptron. O algoritmo de aprednizagem pode ser definido sendo passado por parâmetro.
    Assim, conforme os dois exemplos anteriores, duas abordagens possíveis são a PseudoInversa e a DescidaGradiente.
    '''
    def __init__(self, training_algorithm):
        self.w = None
        self.activation = sign
        self.training_algorithm = training_algorithm
    
    @staticmethod
    def includeBias(X):
        bias = np.ones((X.shape[0], 1))
        Xb = np.concatenate((bias, X), axis=1)
        
        return Xb
        
    def predict(self, X, y=None):
        '''
        Função para predição com dados de testes.
        '''
        Xb = self.includeBias(X)
        a = Xb @ self.w
        ypred = self.activation(a)
        
        return ypred
    
    
    def fit(self, X, y):
        '''
        Treinamento do perceptron.
        '''

        Xb = Perceptron.includeBias(X)
        self.w = self.training_algorithm.getW(Xb, y)

class PerceptronMulticlasse(BaseEstimator, ClassifierMixin):
    def __init__(self, training_algorithm):
        self.w = None
        self.activation = sign
        self.training_algorithm = training_algorithm
          
    @staticmethod
    def includeBias(X):
        bias = np.ones((X.shape[0], 1))
        Xb = np.concatenate((bias, X), axis=1)
        return Xb
        
    def predict(self, X, y=None):
        Xb = self.includeBias(X)
        a = Xb @ self.w
        
        if self.w.shape[1]==1: 
            idx = np.array(a>0,dtype=int).reshape((-1,)) #problema de 2 classes
        else:
            idx = np.argmax(a,axis=1)
        ypred = np.array([self.labels[i] for i in idx]) #pega a coluna de maior probabilidade
        return ypred
    
    def fit(self, X, y):
        yhot = self.encode_labels(y) #não usar as classes qdo o problema é multiclasse, fazer o one-hot-encode
        Xb = PerceptronMulticlasse.includeBias(X) #essas duas linhas são pre tratamentos
        self.w = self.training_algorithm.getW(Xb,yhot)

    def encode_labels(self,y):
        self.labels = list(set(y))
        return label_binarize(y,classes=self.labels)*2-1 

class PerceptronMulticlasseV2(BaseEstimator, ClassifierMixin):
    '''
    Função predict alterada em relação ao de cima.
    '''
    def __init__(self, training_algorithm):
        self.w = None
        self.activation = sign
        self.training_algorithm = training_algorithm
          
    @staticmethod
    def includeBias(X):
        bias = np.ones((X.shape[0], 1))
        Xb = np.concatenate((bias, X), axis=1)
        return Xb
        
    def predict(self, X, y=None):
        Xb = self.includeBias(X)
        a = Xb @ self.w
        
        #>>> ALTERAÇÃO DA DEFINIÇÂO ANTERIOR <<<
        idx = np.argmax(a,axis=1) 
        ypred = np.array([self.labels[i] for i in idx])

        return ypred
    
    def fit(self, X, y):
        yhot = self.encode_labels(y) #não usar as classes qdo o problema é multiclasse, fazer o one-hot-encode
        Xb = PerceptronMulticlasse.includeBias(X) #essas duas linhas são pre tratamentos
        self.w = self.training_algorithm.getW(Xb,yhot)

    def encode_labels(self,y):
        self.labels = list(set(y))
        return label_binarize(y,classes=self.labels)*2-1 