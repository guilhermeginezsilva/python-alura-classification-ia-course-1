# from dados import carregar_buscas
#
# X, Y = carregar_buscas()
# print(X[0])
# print(Y[0])

from collections import Counter
import pandas as pd
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
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

#diferencas = resultado - teste_marcacoes
acertos = (resultado == teste_marcacoes)

#acertos = [d for d in diferencas if d ==0]
#total_de_acertos = len(acertos)

total_de_acertos = sum(acertos) #true = 1 / false = 0 no python
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print("Algoritmo Naive Bayes")

print(taxa_de_acerto)
print(total_de_elementos)

#algoritmo burro para comparacao

print("Algoritmo burro para comparacao - considerando 1")
#acerto_de_um = sum(teste_marcacoes) # len(teste_marcacoes[teste_marcacoes==1]) ou list(teste_marcacoes).count('sim') se fosse uma string
acerto_de_um = list(teste_marcacoes).count('sim')
print(100.0 * acerto_de_um/len(teste_marcacoes))
print(len(teste_marcacoes))

print("Algoritmo burro para comparacao - considerando 0")
# acerto_de_zero = len(teste_marcacoes) - acerto_de_um # len(teste_marcacoes[teste_marcacoes==0]) ou list(teste_marcacoes).count('nao') se fosse uma string
acerto_de_zero = list(teste_marcacoes).count('nao')
print(100.0 * acerto_de_zero/len(teste_marcacoes))
print(len(teste_marcacoes))

acerto_base = max(Counter(teste_marcacoes).values())

taxa_de_acerto_base = 100.0 * acerto_base/len(teste_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)



