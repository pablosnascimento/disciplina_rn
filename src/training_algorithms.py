from src.cost_algorithms import WidrowHoff
from src.interfaces import TrainingAlgorithm
import numpy as np

class PseudoInversa(TrainingAlgorithm):
    '''
    VERIFICAR: função de calculo do w não esta funcionando devidamente.
    '''
    def __init__(self, regularization=0):
        self.regularization = regularization
    
    def getW(self, X, y):
        if self.regularization == 0:
            return np.linalg.pinv(X) @ y
        return np.linalg.inv(X.T @ X + np.eye(X.shape[1])*self.regularization) @ X.T @ y

class DescidaGradiente(TrainingAlgorithm):
    '''
    Abordagem Descida de Gradiente para treinamento da rede.
    '''
    def __init__(self,max_iter=100,learning_rate=0.02, regularization=0, cost=WidrowHoff()):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        #evitar overfiting. Funciona como uma "taxa de esquecimento" no aprendizado
        #quanto maior a regularização, mais esquecimento do w anterior eu tenho na nova iteração
        self.regularization = regularization
        self.cost = cost
        self.custos = []

    def getW(self,X,y):
        """
        getW Versão Padrao
        """
        w = np.random.uniform(-1,1,size=X.shape[1])
        
        for j in range(self.max_iter):
            ypred = X @ w
                        
            c = self.cost.custo(y, ypred) 
            #custos
            self.custos.append(c)

            if c == 0:
                break            

            #aplicando a regularização..
            w *= 1 - self.learning_rate * self.regularization
            w += (X.T @ self.cost.gradiente(y, ypred)) * self.learning_rate
            
        return w

        '''
        getW com alterações para PerceptronMulticlasse
        '''
        # w =np.random.uniform(-1,1,size=(X.shape[1],y.shape[1])) #começa com um vetor de pesos aleatórios
        # for _ in range(self.max_iter):
        #     ypred = X @ w
        #     c = self.cost.custo(y, ypred) 
        #     self.custos.append(c)
        #     if c==0:
        #         break
        #     w *= 1-self.regularization*self.learning_rate
        #     w += (X.T @ self.cost.gradiente(y, ypred))*self.learning_rate
        # return w