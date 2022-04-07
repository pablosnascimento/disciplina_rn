from abc import ABC, abstractmethod, abstractstaticmethod

class ICusto(ABC):
    '''
    Interface genérica para definição de uma classe de custo.
    Necessário implementar métodos "custo" e "gradiente".
    '''
    @abstractstaticmethod
    def custo(self, y, ypred):
        pass

    @abstractstaticmethod
    def gradiente(self, y, ypred):
        pass


class TrainingAlgorithm(ABC):
    '''
    Interface genérica para definição de um algoritmo de treinamento base para o Perceptron.
    Necessário implementar método getW.
    '''
    @abstractmethod
    def getW(self, X, y):
        pass