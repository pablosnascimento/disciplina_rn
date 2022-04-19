import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

################### FUNCOES INTERNAS AO PERCPETRON E SEUS ALGORTIMOS ##############
def sign(a):
    '''
    Função 'sinal': retorna 1 ou -1 conforme valores de a, ou seja, é uma binarização dos valores do vetor para apenas -1 e 1.
    '''
    return (a >= 0)*2-1

def accuracy(y, ypred):
    '''
    Função para encontrar a acurácia do resultado encontrado.
    '''
    return sum(y==ypred)/len(y)
####################################################################################

################### GERACAO DE DATASET ###################
def CriaDataSetClassificacao(n=20, slop=[2,1], intercept=-0.4): 
    '''
    Cria um dataser para classificação binária com n amostras.
    '''

    X = np.random.uniform(size=(n,2))
    AUX = np.multiply(X,slop)-[0, intercept]
    y = (AUX[:,0] > X[:,1])*2-1
    
    return X, y

def CriaDataSetRegressao(n=20, slop=0.5, intercept=0.2): 
    '''
    Cria um dataser para regressão n amostras com uma dimensão.
    '''

    X = np.random.uniform(size=(n,1)) #n,1 apenas uma dimensão
    AUX = np.random.rand(n,1)-0.5
    
    y = X*slop + intercept + AUX*0.1
    y = y.flatten()
    
    return X, y

def CriaDataSetRegularizacao(n=20, slop=[2,1], intercept=-0.4, dummy_features=3): 
    '''
    Dataset para ilustrar regularização no treinamento.
    Criaremos uma base que possui algum ruído proposital para avaliar o comportamento do treinamento com ele.
    '''
    
    X = np.random.uniform(size=(n,dummy_features))
    AUX = np.multiply(X[:,:2],slop)-[0, intercept] # equivalente @, multiplicação de matrizes
    y = np.array(AUX[:,0] > AUX[:,1], dtype=int)*2-1
    
    return X, y

def criaDatasetMulticlasse(n=1000,n_classes=4):
    '''
    Gera uma base de dados para problemas multiclasse (n_classes conforme parâmetro).
    '''
    X,y = make_blobs(n_samples=n,centers=n_classes,center_box=(0,1),cluster_std=0.02)
    return X,y

def criaDatasetXOR(n=1000):
    '''
    Cria dataset XOR, não linearmente separável.
    Será necessária uma camada oculta para resolver. Próxima aula
    '''
    X,y = make_blobs(n_samples=n,centers=[[0,0],[1,0],[1,1],[0,1]],cluster_std=0.1)
    y = np.array(y%2,dtype=int)
    return X,y
######################################################################################

################################# PLOTAGEM DE GRÁFICOS ###############################
def plotDatasetRegressao(X, y):
    '''
    Função para plotar o dataset para a regressão, pois difere devido a apenas
    uma dimensão do X.
    '''
    
    plt.plot(X[:,0], y, "o", alpha=0.3)

def plotDataSet(X,y):
    '''
    Exibe gráfico com os dados da base gerada dividindo em cores distintas os grupos diferentes.
    '''
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    for k in set(y):
        #print("k=",k)
        plt.plot(X[:,0][y==k],
                 X[:,1][y==k],
                 "o",alpha=0.3)

def plotHiperplano(X,y,vetor, intercept=0):
    '''
    Plota a linha que divide a base de dados sobre o gráfico.
    '''
    x0min = min(X[:,0])
    x0max = max(X[:,0])
    
    xs = np.linspace(x0min, x0max, num=2)
    #separador do hiperplano entre duas classificações pode ser 
    #encontrada conforme calculo abaixo:
    ys = (-vetor[0]/vetor[1])*xs-intercept/vetor[1]
    plt.plot(xs,ys)

def PlotCusto(custos):
    '''
    Recebe um vetor de custos (decimais) e plota o gráfico de iterações x custo do aprendizado para avaliação do quão rápido a rede aprendeu a classificação.
    '''
    if (len(custos) == 0):
        print('ERRO: Modelo ainda não treinado para avaliar o custo!')
        return
    
    x = [i for i in range(1,len(custos) + 1)]

    plt.xlabel('Nº Iterações')
    plt.ylabel('Custo')
    plt.plot(x, custos, "-")
    plt.show()
######################################################################################