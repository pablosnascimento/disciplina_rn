from src.interfaces import ICusto
import numpy as np
from scipy.special import expit

from src.util import sign

class WidrowHoff(ICusto):
    '''
    Esta função de custo sofre com a distância de pontos distantes do hiperplano.
    Quem está distante influencia o coeficiente final.
    '''
    @staticmethod
    def custo(y, ypred):
        return np.sum((y - ypred)**2)
    
    @staticmethod
    def gradiente(y, ypred):
        return y - ypred


class SmoothedSurrogate(ICusto):
    '''
    Função de custo cujo gradiente utiliza o mesmo método do PLA Perseptron inicial (Aula1), mas
    a função de custo é calculada de forma diferente.

    custo = - y * ypred
    '''
    @staticmethod
    def custo(y, ypred):
        return np.sum(np.maximum(np.zeros(y.shape), -y*ypred))
    
    @staticmethod
    def gradiente(y, ypred):
        return y - sign(ypred)


class LogLikehood(ICusto):
    '''
    
    '''
    @staticmethod
    def custo(y, ypred):
        return np.sum(np.log(1 + np.exp(-y * ypred)))
    
    @staticmethod
    def gradiente(y, ypred):
        return y - (expit(ypred)*2 -1 )


class HingeLoss(ICusto):
    '''
    Função de custo tende a considerar quem está mais próximo do hiperplano.
    custo -> 1 - y * ypred
    gradiente -> considera os erros negativos com maior rigor que os positivos, de forma a penalizar
    o algoritmo quando erra muito negativamente (errosmarginais) 
    '''
    @staticmethod
    def custo(y, ypred):
        return np.sum(np.maximum(np.zeros(y.shape), 1 - (y * ypred)))
    
    @staticmethod
    def gradiente(y, ypred):
        #filtro para encontrar os erros marginais
        errosmarginais = (y * ypred) < 1
        ymarginais = np.copy(y)
        
        #zerando todos os erros que não são marginais
        #invertendo os indices, 
        ymarginais[~errosmarginais] = 0 

        return ymarginais