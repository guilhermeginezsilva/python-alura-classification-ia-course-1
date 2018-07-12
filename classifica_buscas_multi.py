# from dados import carregar_buscas
#
# X, Y = carregar_buscas()
# print(X[0])
# print(Y[0])

from collections import Counter
import pandas as pd

def fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    #diferencas = resultado - teste_marcacoes
    acertos = (resultado == teste_marcacoes)

    #acertos = [d for d in diferencas if d ==0]
    #total_de_acertos = len(acertos)

    total_de_acertos = sum(acertos) #true = 1 / false = 0 no python
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do {0}: {1}".format(modelo.__class__.__name__, taxa_de_acerto)
    print(msg)

df = pd.read_csv('buscas_str.csv') #dataframe
print(df)

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df #pd.get_dummies(Y)[1]

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.9

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
fit_and_predict(modelo,treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modelo = AdaBoostClassifier()
fit_and_predict(modelo,treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

#algoritmo burro para comparacao
acerto_base = max(Counter(teste_marcacoes).values())

taxa_de_acerto_base = 100.0 * acerto_base/len(teste_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d" % len(teste_dados))



