from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABC, abstractmethod, abstractstaticmethod
import numpy as np

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