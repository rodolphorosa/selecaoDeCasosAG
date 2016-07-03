import re, math, random, numpy as np
from pprint import pprint
from random import randint, random, sample

np.set_printoptions(precision=2)

ALFA = 3.0
BETA = 1.0
TAXA_CROSSOVER = 0.5
TAXA_MUTACAO = 0.2

# Calcula do tempo do suite 
# O tempo eh a soma do tempo de execucao de todos os casos que compoem o suite 
def tempoSuite(suite, casos):
	tempos = [ casos[i,1] for i in range(suite.shape[0]) if suite[i] == 1 ]

	if tempos == []:
		return 0
	return sum(tempos)

def qtdeExecucoesSuite(suite, casos):
	execucoes = [ casos[i,3] for i in range(suite.shape[0]) if suite[i] == 1 ]

	if execucoes == []:
		return 0
	return sum(execucoes)

# Calcula a importancia do suite. 
def importanciaSuite(suite, casos):
	casos_suite = [ casos[i,4] for i in range(suite.shape[0]) if suite[i] == 1 ]
	if casos_suite == []:
		return 0
	media = np.mean(casos_suite)
	return media

# Verifica a viabilidade de adicionar um caso ao suite dada a restricao de tempo. 
def casoViavel(caso, suite, casos, restricao):
	if (tempoSuite(suite, casos) + caso[1]) <= (sum(casos[:,1]) * restricao):
		return True
	else:
		return False

# Verifica a viabilidade da solucao 
# Uma solucao eh viavel se todos os precedentes dos casos do suite tambem estiverem no suite
# E se o tempo de execucao do suite for menor ou igual a restricao de tempo 
def solucaoViavel(suite, casos, restricao_tempo):
	casos_suite = [i for i in range(0, suite.shape[0]) if suite[i] == 1]

	for caso in casos_suite: # Para cada caso do suite
		precedente = casos[caso][2] # Verifica o precedente
		if precedente != 0 and suite[precedente-1] == 0: # Se o caso tiver precedente e ele nao estiver no suite
			return False # A solucao nao eh viavel

	if tempoSuite(suite, casos) > (sum(casos[:,1]) * restricao_tempo): # Se o tempo do suite for maior que o permitido pela restricao
		return False # A solucao nao eh valida
	
	return True

# Retorna o caso precedente de um caso de entrada. 
def precedenteCaso(caso, casos):
	for idx in range(casos.shape[0]):
		if casos[idx,0] == caso[2]:
			return idx
	return None

def iniciarPopulacao(populacao, casos, restricao):
	t_populacao = populacao.shape[0]
	n_casos 	= casos.shape[0]

	for i in range(t_populacao): # Para cada caso da populacao 
		suite = np.zeros(n_casos) # Gera um suite vazio 
		while True: # Enquanto nao terminar 
			rand = randint(0, n_casos-1) # seleciona um caso aleatorio 
			caso = casos[rand] 
			precedente = caso[2] 			
			if suite[rand] == 1: # Se o caso ja estiver no suite, 
				continue # reinicia o processo 			
			if casoViavel(caso, suite, casos, restricao): # Se for viavel adicionar o caso ao suite 
				if precedente == 0: # Se o caso nao tiver precedente, 
					suite[rand] = 1 # ele eh adicionado ao suite 
				else: # Caso o caso possua um precedente, 
					cpy_suite = np.copy(suite)
					cpy_suite[rand] = 1 # adiciona-se o caso e seu precedente ao suite, 
					while True:	# e enquanto for viavel, o precedente deste tambem eh adicionado 
						if precedente == 0:
							suite = cpy_suite
							break
						if casoViavel(casos[precedente-1], cpy_suite, casos, restricao):
							cpy_suite[precedente-1] = 1
							precedente = casos[precedente-1][2]
						else:
							break
			else:
				break
		populacao[i] = suite

# Verifica se um numero aleatorio eh menor que a probabilidade dada como entrada.
def flip(probabilidade):
	rand = random()
	return rand < probabilidade

# Calcula a fitness (avaliacao) de um individuo (suite)
def fitness(suite, casos):
	fit = ALFA * qtdeExecucoesSuite(suite, casos) + BETA * importanciaSuite(suite, casos)
	return fit

# Seleciona um individuo para crossover (atraves de competicao)
def selecionar(populacao, casos):
	individuos 	= sample(range(populacao.shape[0] - 1), 3) # Seleciona tres casos aleatorios 
	fitnesses 	= [fitness(populacao[ind], casos) for ind in individuos] # Verifica a funcao de avaliacao para cada caso 
	selecionado = individuos[fitnesses.index(max(fitnesses))] # Seleciona o caso com maior fitness 
	return selecionado

def crossover(pais):

	tamanho_cromossomo = pais[0].shape[0]

	if flip(TAXA_CROSSOVER):
		#ponto_corte = 50
		ponto_corte = randint(0, tamanho_cromossomo-1)
	else:
		ponto_corte = tamanho_cromossomo - 1

	filho0 = np.concatenate((pais[0][:ponto_corte], pais[1][ponto_corte:]))
	filho1 = np.concatenate((pais[1][:ponto_corte], pais[0][ponto_corte:]))

	return (filho0,filho1)

def mutacao(suite, casos, restricao_tempo):

	casos_suite = [i for i in range(0, suite.shape[0]) if suite[i] == 1] # Seleciona os casos do suite 
		
	# Casos cujo precedente nao esta no suite 
	# Esses casos podem ser excluidos ou seu precedente, adicionado
	candidatos 	= []  
	for caso in casos_suite: # Para cada caso 
		precedente = casos[caso][2] # Verifica o seu precedente 
		if precedente != 0 and suite[precedente-1] == 0: # Se o caso tiver precedente e ele nao estiver no suite 
			candidatos.append(caso) # Adiciona o caso a lista 
	
	for caso in candidatos:
		suite[caso] = 0

	if tempoSuite(suite, casos) > (sum(casos[:,1]) * restricao_tempo):
		menor = menor_caso(suite, casos)
		suite[menor] = 0

def menor_caso(suite, casos):
	tempos = [ casos[i, 1] if suite[i] == 1 else 101 for i in range(suite.shape[0]) ]
	menor_caso = min(tempos)
	return tempos.index(menor_caso)

def get_melhor_solucao(populacao, casos):
	fitness_populacao = [ fitness(populacao[i], casos) for i in range(populacao.shape[0]) ]
	return fitness_populacao.index(max(fitness_populacao))

def imprimirSuite(suite,casos):
	for i in range(suite.shape[0]):
		if suite[i] == 1:
			print(casos[i])

# Executa o algoritmo genetico dadas as configuracoes de entrada
def ag(tamanho_populacao, restricao_tempo, max_iteracoes):
	dataset = open("DATASET", 'r')
	
	header = dataset.readline()
	n_casos = int(re.findall("[0-9]+", header)[0])

	casos = np.zeros((n_casos, 5), dtype=np.int)
	rows = [row for row in dataset]

	for row in range(len(rows)):
		data = rows[row].split()
		casos[row][0] = data[0]
		casos[row][1] = data[1]
		casos[row][2] = data[2]
		casos[row][3] = data[3]
		casos[row][4] = data[4]

	populacao_t0 = np.zeros((tamanho_populacao, n_casos), dtype=np.int)
	iniciarPopulacao(populacao_t0, casos, restricao_tempo) 

	# print "Melhor solucao gerada inicialmente"
	# melhor_solucao = get_melhor_solucao(populacao_t0, casos)
	# imprimirSuite(populacao_t0[melhor_solucao], casos)
	# print "Funcao objetivo: ", fitness(populacao_t0[melhor_solucao], casos) 

	t = 0

	while t < max_iteracoes:
		populacao_t1 = np.zeros((tamanho_populacao, n_casos), dtype=np.int) 

		selecionados = []
		for i in range(0, 100):
			selecionados.append(selecionar(populacao_t0, casos))

		for i in range(0, 100, 2):
			filhos		 		= crossover((populacao_t0[selecionados[i]], populacao_t0[selecionados[i+1]])) 
			populacao_t1[i] 	= filhos[0] 
			populacao_t1[i+1] 	= filhos[1] 

		for i in range(0, 100): 
			while not solucaoViavel(populacao_t1[i], casos, restricao_tempo):
				mutacao(populacao_t1[i], casos, restricao_tempo)

		populacao_t0 = populacao_t1
		
		t = t + 1

	print "\nMelhor solucao gerada pelo algoritmo"
	melhor_solucao = get_melhor_solucao(populacao_t0, casos)
	imprimirSuite(populacao_t0[melhor_solucao], casos)
	print "Funcao objetivo: ", fitness(populacao_t0[melhor_solucao], casos) 

	return

if __name__ == "__main__":
	ag(100, 0.8, 10) 