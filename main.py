
import argparse
from src.cost_algorithms import HingeLoss, LogLikehood, SmoothedSurrogate, WidrowHoff
from src.perceptron import Perceptron
from src.training_algorithms import DescidaGradiente, PseudoInversa
import src.util as util
import matplotlib.pyplot as plt

def read_command_line_args():
    '''
    Função que lê os argumentos passados na linha de comando e retorna a variável args com os parâmetros.
    '''
    parser = argparse.ArgumentParser(
        description='Perceptron e suas derivações - custos e algoritmos de aprendizagem.')

    parser.add_argument('-a', '--algorithm', default='DescidaGradiente', choices=["DescidaGradiente", "PseudoInversa"], help="Algoritmo de aprendizado para uso no Perceptron.")
    parser.add_argument('-c', '--cost', default='HingeLoss', choices=["WidrowHoff", "SmoothedSurrogate", "LogLikehood", "HingeLoss"], help='Função de custo para treinamento da rede.')
    
    parser.add_argument('-m', '--max_iterations', type=int, default=100, help='Número máximo de iterações no treinamento.')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.02, help='Taxa de aprendizado (variação pequena de 0.02 a 0.07, fora disso tende a divergir).')
    parser.add_argument('-r', '--regularization', type=float, default=0, help='Taxa de regularização.')

    args = parser.parse_args()
    return args

def build_perceptron(args):
    '''
    Instancia a classe Perceptron conforme especificações parametrizadas.
    '''
    #função de custo
    funcao_custo = None
    if args.cost == "WidrowHoff":
        funcao_custo = WidrowHoff()
    elif args.cost == "SmoothedSurrogate":
        funcao_custo = SmoothedSurrogate()
    elif args.cost == "LogLikehood":
        funcao_custo = LogLikehood()
    elif args.cost == "HingeLoss":
        funcao_custo = HingeLoss()
    else:
        raise NotImplementedError("Erro: Função de custo escolhida não foi encontrada.")

    if args.algorithm == "DescidaGradiente":
        return Perceptron(training_algorithm=DescidaGradiente(  max_iter=args.max_iterations, 
                                                                learning_rate=args.learning_rate, 
                                                                regularization=args.regularization,
                                                                cost=funcao_custo))
    elif args.algorithm == "PseudoInversa":
        raise Perceptron(training_algorithm=PseudoInversa(regularization=args.regularization))
    else:
        raise NotImplementedError("Erro: Algoritmo de treinamento escolhido não encontrado.")

def main():

    args = read_command_line_args()
    percep = build_perceptron(args)

    #cria uma base de dados
    X, y = util.CriaDataSetClassificacao(n=1000)

    '''
    from sklearn.model_selection import train_test_split

    #criando um dataset para exemplo multiclasse
    X,y = criaDatasetMulticlasse()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.9)
    '''

    #treina a rede
    percep.fit(X,y)
    ypred = percep.predict(X)
    print('ACURÁCIA NO TESTE > ' + args.algorithm + " + " + args.cost + ":", util.accuracy(y=y, ypred=ypred))

    #plotra os resultados
    util.plotDataSet(X, y)
    util.plotHiperplano(X,y,percep.w[1:], percep.w[0])
    plt.show()

    util.PlotCusto(percep.training_algorithm.custos)


if __name__ == "__main__":
    main()