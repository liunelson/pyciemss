# Adapted/translated from https://github.com/scabini/COmplexVID-19 and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7659518/

# Load dependencies
import os
import sys
from multiprocessing import Pool
import networkx as nx
import model
from model import isolate_node
import numpy as np
import random
from collections import Counter
from comunities import *
# from dynamics import *
from dynamics_OTM import *
import random
import matplotlib.pyplot as plt
import pickle
from time import perf_counter
import matplotlib.ticker as ticker

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Define parameters, key demographic info
location = "Sao Carlos"
#beta_list = [0.2, 0.25, 0.3, 0.35]
beta_global = 0.35 # baseline transmission rate with no intervention (best fit to Brazil data)
n = 100000 # number of nodes, max 100K due to runtime, N * population_scale_factor = total_population
population_scale_factor = 57
repetitions = 10 #100 # simulation repetitions, final results are averaged

primeiro_infectado = "Fevereiro 26" # date of first infection
infected_t0 = 1 # initial infected population
acoes = ["Março 24"] # list of intervention dates
ano = {'Janeiro 1': 0, 'Janeiro 2': 1, 'Janeiro 3': 2, 'Janeiro 4': 3, 'Janeiro 5': 4, 'Janeiro 6': 5, 'Janeiro 7': 6, 'Janeiro 8': 7, 'Janeiro 9': 8, 'Janeiro 10': 9, 'Janeiro 11': 10, 'Janeiro 12': 11, 'Janeiro 13': 12, 'Janeiro 14': 13, 'Janeiro 15': 14, 'Janeiro 16': 15, 'Janeiro 17': 16, 'Janeiro 18': 17, 'Janeiro 19': 18, 'Janeiro 20': 19, 'Janeiro 21': 20, 'Janeiro 22': 21, 'Janeiro 23': 22, 'Janeiro 24': 23, 'Janeiro 25': 24, 'Janeiro 26': 25, 'Janeiro 27': 26, 'Janeiro 28': 27, 'Janeiro 29': 28, 'Janeiro 30': 29, 'Janeiro 31': 30, 'Fevereiro 1': 31, 'Fevereiro 2': 32, 'Fevereiro 3': 33, 'Fevereiro 4': 34, 'Fevereiro 5': 35, 'Fevereiro 6': 36, 'Fevereiro 7': 37, 'Fevereiro 8': 38, 'Fevereiro 9': 39, 'Fevereiro 10': 40, 'Fevereiro 11': 41, 'Fevereiro 12': 42, 'Fevereiro 13': 43, 'Fevereiro 14': 44, 'Fevereiro 15': 45, 'Fevereiro 16': 46, 'Fevereiro 17': 47, 'Fevereiro 18': 48, 'Fevereiro 19': 49, 'Fevereiro 20': 50, 'Fevereiro 21': 51, 'Fevereiro 22': 52, 'Fevereiro 23': 53, 'Fevereiro 24': 54, 'Fevereiro 25': 55, 'Fevereiro 26': 56, 'Fevereiro 27': 57, 'Fevereiro 28': 58, 'Fevereiro 29': 59, 'Março 1': 60, 'Março 2': 61, 'Março 3': 62, 'Março 4': 63, 'Março 5': 64, 'Março 6': 65, 'Março 7': 66, 'Março 8': 67, 'Março 9': 68, 'Março 10': 69, 'Março 11': 70, 'Março 12': 71, 'Março 13': 72, 'Março 14': 73, 'Março 15': 74, 'Março 16': 75, 'Março 17': 76, 'Março 18': 77, 'Março 19': 78, 'Março 20': 79, 'Março 21': 80, 'Março 22': 81, 'Março 23': 82, 'Março 24': 83, 'Março 25': 84, 'Março 26': 85, 'Março 27': 86, 'Março 28': 87, 'Março 29': 88, 'Março 30': 89, 'Março 31': 90, 'Abril 1': 91, 'Abril 2': 92, 'Abril 3': 93, 'Abril 4': 94, 'Abril 5': 95, 'Abril 6': 96, 'Abril 7': 97, 'Abril 8': 98, 'Abril 9': 99, 'Abril 10': 100, 'Abril 11': 101, 'Abril 12': 102, 'Abril 13': 103, 'Abril 14': 104, 'Abril 15': 105, 'Abril 16': 106, 'Abril 17': 107, 'Abril 18': 108, 'Abril 19': 109, 'Abril 20': 110, 'Abril 21': 111, 'Abril 22': 112, 'Abril 23': 113, 'Abril 24': 114, 'Abril 25': 115, 'Abril 26': 116, 'Abril 27': 117, 'Abril 28': 118, 'Abril 29': 119, 'Abril 30': 120, 'Maio 1': 121, 'Maio 2': 122, 'Maio 3': 123, 'Maio 4': 124, 'Maio 5': 125, 'Maio 6': 126, 'Maio 7': 127, 'Maio 8': 128, 'Maio 9': 129, 'Maio 10': 130, 'Maio 11': 131, 'Maio 12': 132, 'Maio 13': 133, 'Maio 14': 134, 'Maio 15': 135, 'Maio 16': 136, 'Maio 17': 137, 'Maio 18': 138, 'Maio 19': 139, 'Maio 20': 140, 'Maio 21': 141, 'Maio 22': 142, 'Maio 23': 143, 'Maio 24': 144, 'Maio 25': 145, 'Maio 26': 146, 'Maio 27': 147, 'Maio 28': 148, 'Maio 29': 149, 'Maio 30': 150, 'Maio 31': 151, 'Junho 1': 152, 'Junho 2': 153, 'Junho 3': 154, 'Junho 4': 155, 'Junho 5': 156, 'Junho 6': 157, 'Junho 7': 158, 'Junho 8': 159, 'Junho 9': 160, 'Junho 10': 161, 'Junho 11': 162, 'Junho 12': 163, 'Junho 13': 164, 'Junho 14': 165, 'Junho 15': 166, 'Junho 16': 167, 'Junho 17': 168, 'Junho 18': 169, 'Junho 19': 170, 'Junho 20': 171, 'Junho 21': 172, 'Junho 22': 173, 'Junho 23': 174, 'Junho 24': 175, 'Junho 25': 176, 'Junho 26': 177, 'Junho 27': 178, 'Junho 28': 179, 'Junho 29': 180, 'Junho 30': 181, 'Julho 1': 182, 'Julho 2': 183, 'Julho 3': 184, 'Julho 4': 185, 'Julho 5': 186, 'Julho 6': 187, 'Julho 7': 188, 'Julho 8': 189, 'Julho 9': 190, 'Julho 10': 191, 'Julho 11': 192, 'Julho 12': 193, 'Julho 13': 194, 'Julho 14': 195, 'Julho 15': 196, 'Julho 16': 197, 'Julho 17': 198, 'Julho 18': 199, 'Julho 19': 200, 'Julho 20': 201, 'Julho 21': 202, 'Julho 22': 203, 'Julho 23': 204, 'Julho 24': 205, 'Julho 25': 206, 'Julho 26': 207, 'Julho 27': 208, 'Julho 28': 209, 'Julho 29': 210, 'Julho 30': 211, 'Julho 31': 212, 'Agosto 1': 213, 'Agosto 2': 214, 'Agosto 3': 215, 'Agosto 4': 216, 'Agosto 5': 217, 'Agosto 6': 218, 'Agosto 7': 219, 'Agosto 8': 220, 'Agosto 9': 221, 'Agosto 10': 222, 'Agosto 11': 223, 'Agosto 12': 224, 'Agosto 13': 225, 'Agosto 14': 226, 'Agosto 15': 227, 'Agosto 16': 228, 'Agosto 17': 229, 'Agosto 18': 230, 'Agosto 19': 231, 'Agosto 20': 232, 'Agosto 21': 233, 'Agosto 22': 234, 'Agosto 23': 235, 'Agosto 24': 236, 'Agosto 25': 237, 'Agosto 26': 238, 'Agosto 27': 239, 'Agosto 28': 240, 'Agosto 29': 241, 'Agosto 30': 242, 'Agosto 31': 243, 'Setembro 1': 244, 'Setembro 2': 245, 'Setembro 3': 246, 'Setembro 4': 247, 'Setembro 5': 248, 'Setembro 6': 249, 'Setembro 7': 250, 'Setembro 8': 251, 'Setembro 9': 252, 'Setembro 10': 253, 'Setembro 11': 254, 'Setembro 12': 255, 'Setembro 13': 256, 'Setembro 14': 257, 'Setembro 15': 258, 'Setembro 16': 259, 'Setembro 17': 260, 'Setembro 18': 261, 'Setembro 19': 262, 'Setembro 20': 263, 'Setembro 21': 264, 'Setembro 22': 265, 'Setembro 23': 266, 'Setembro 24': 267, 'Setembro 25': 268, 'Setembro 26': 269, 'Setembro 27': 270, 'Setembro 28': 271, 'Setembro 29': 272, 'Setembro 30': 273, 'Outubro 1': 274, 'Outubro 2': 275, 'Outubro 3': 276, 'Outubro 4': 277, 'Outubro 5': 278, 'Outubro 6': 279, 'Outubro 7': 280, 'Outubro 8': 281, 'Outubro 9': 282, 'Outubro 10': 283, 'Outubro 11': 284, 'Outubro 12': 285, 'Outubro 13': 286, 'Outubro 14': 287, 'Outubro 15': 288, 'Outubro 16': 289, 'Outubro 17': 290, 'Outubro 18': 291, 'Outubro 19': 292, 'Outubro 20': 293, 'Outubro 21': 294, 'Outubro 22': 295, 'Outubro 23': 296, 'Outubro 24': 297, 'Outubro 25': 298, 'Outubro 26': 299, 'Outubro 27': 300, 'Outubro 28': 301, 'Outubro 29': 302, 'Outubro 30': 303, 'Outubro 31': 304, 'Novembro 1': 305, 'Novembro 2': 306, 'Novembro 3': 307, 'Novembro 4': 308, 'Novembro 5': 309, 'Novembro 6': 310, 'Novembro 7': 311, 'Novembro 8': 312, 'Novembro 9': 313, 'Novembro 10': 314, 'Novembro 11': 315, 'Novembro 12': 316, 'Novembro 13': 317, 'Novembro 14': 318, 'Novembro 15': 319, 'Novembro 16': 320, 'Novembro 17': 321, 'Novembro 18': 322, 'Novembro 19': 323, 'Novembro 20': 324, 'Novembro 21': 325, 'Novembro 22': 326, 'Novembro 23': 327, 'Novembro 24': 328, 'Novembro 25': 329, 'Novembro 26': 330, 'Novembro 27': 331, 'Novembro 28': 332, 'Novembro 29': 333, 'Novembro 30': 334, 'Dezembro 1': 335, 'Dezembro 2': 336, 'Dezembro 3': 337, 'Dezembro 4': 338, 'Dezembro 5': 339, 'Dezembro 6': 340, 'Dezembro 7': 341, 'Dezembro 8': 342, 'Dezembro 9': 343, 'Dezembro 10': 344, 'Dezembro 11': 345, 'Dezembro 12': 346, 'Dezembro 13': 347, 'Dezembro 14': 348, 'Dezembro 15': 349, 'Dezembro 16': 350, 'Dezembro 17': 351, 'Dezembro 18': 352, 'Dezembro 19': 353, 'Dezembro 20': 354, 'Dezembro 21': 355, 'Dezembro 22': 356, 'Dezembro 23': 357, 'Dezembro 24': 358, 'Dezembro 25': 359, 'Dezembro 26': 360, 'Dezembro 27': 361, 'Dezembro 28': 362, 'Dezembro 29': 363, 'Dezembro 30': 364, 'Dezembro 31': 365}
ano_rev = inv_map = {v: k for k, v in ano.items()}
days = 300 - ano[primeiro_infectado] # days to simulate, beginning from date of first infection
begin = ano[primeiro_infectado]
acoes1 = [ano[i]-begin for i in acoes]

# Age distribution, age_dist must sum to 1
age_dist = np.zeros((6))
age_dist[0] = 0.18 # 0-13  - school
age_dist[1] = 0.06 # 14-17 - school
age_dist[2] = 0.11 # 18-24 - work
age_dist[3] = 0.23 # 25-39 - work
age_dist[4] = 0.26 # 40-59 - work
age_dist[5] = 0.16 # 60+

# Distribution of family sizes, fam_structure must sum to 1
fam_structure = np.zeros((10))
fam_structure[0] = 0.12 # 1 person
fam_structure[1] = 0.22 # 2 people
fam_structure[2] = 0.25 # 3 people
fam_structure[3] = 0.21 # 4 people
fam_structure[4] = 0.11 # 5 people
fam_structure[5] = 0.05 # 6 people
fam_structure[6] = 0.02 # 7 people
fam_structure[7] = 0.01 # 8 people
fam_structure[8] = 0.005 # 9 people
fam_structure[9] = 0.005 # 10 people

# Church attendance and use of public transportation
qtde_religiao = 0.4 # fraction of the population that goes to church 1x/week
qtde_transporte = 0.36 # fraction of the population that uses public transportation
tempo_transporte = 1.2  # number of hours spent using public transportation per day
home_isolation = 'home - total' # total home isolation: only home contacts, partial: home and random
hospital_isolation = 'hospital - total' # total hospital isolation: completely isolate the person, partial: include random links only with others in the hospital

# Model layers
layer_names = ['casas', 'aleatorio', 'trabalho', 'transporte', 'escolas', 'igrejas'] # names of the layers: houses, random interactions, work, public transportation, school, church
layers_0 = ['casas', 'aleatorio', 'trabalho', 'transporte', 'escolas', 'igrejas'] # initial model layers (assuming starting in SP quarantine)
layers_tirar = [["escolas", "igrejas"], ["trabalho"]] # (list of lists) layers to remove in response to each action
layers_por = [[], []] # list of lists, layers to return for each action

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# def get_abm_parameters(beta_list, which_beta):
#     beta_global = beta_list[which_beta]
parameters = dict(
    seed=999666, # fixed seed for random operations
    age_dist = age_dist,
    fam_structure = fam_structure, #distribuicao de tamanhos de familias
    tempo_transporte = tempo_transporte,
    qtde_transporte = qtde_transporte,
    qtde_religiao = qtde_religiao,
    layers_0 = layers_0, #camadas iniciais do modelo
    acoes = acoes1,
    layers_tirar = layers_tirar,
    layers_por = layers_por,
    n_nodes = n, #quantidade de pessoas usadas pra estimar as % da epidemia
    #as probabilidade reais são dinamicas e dependem de varias coisas, que são
    #definidas la dentro da criaçao das comunidades. Esse valor aqui é "quanto
    #considerar desses valores: 0-> anula, 1-> original, 0.5-> metade, 2-> dobro
    prob_home = beta_global,       #original> familia toda, 3hrs/dia. Cada camada removida aumenta a interação em casa em 25%
    prob_random = beta_global,     #original> 1 proximo, 1hrs/semana, todos tem chance de ter de 1 a 10 conexoes aleatorias
    prob_work = beta_global,       #original> 4 proximos/tamanho da empresa, 6hrs/dia, 5 dias/semana, todos de 18 a 59 anos.
    prob_transport = beta_global,  #original> 5 proximos/tamanho do veiculo, 1.2hrs/dia, 50% da população (aleatoria)
    prob_school = beta_global,     #original> 4 proximos/tamanho da sala, 5hrs/dia, 5 dias/semana, toda população de 0 a 17 anos
    prob_religion = beta_global,   #original> 6 proximos/tamanho da igreja, 2hrs/semana, 40% da populaçao (aleatorio)
    verbose=False       #printar ou nao as informações durante construção e simulação
)

    # return beta_global, parameters

# print(get_abm_parameters(beta_list, 0))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def fun(data):
    return " ".join([item for var in data for item in var])

def analyze(i):
    parameters['seed'] = i
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])
    count=[]
    G = model.createGraph(parameters)
    G, count = simulate(G, parameters, infected_t0=infected_t0, days=days, hospital_isolation=hospital_isolation, home_isolation=home_isolation)
    G = []
    count = np.true_divide(count, n)
    return count

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def simulate_the_abm(parameters, acoes, location, beta_global, repetitions):

    if __name__ == '__main__':
        print("Initial layers: ", parameters["layers_0"])
        print("Intervention date(s): ", parameters["acoes"])
        print("Layers inserted: ", parameters["layers_por"])
        print("Layers removed: ", parameters["layers_tirar"])

        file = 'experimentos/' + location + '_REALISTIC_-' + str(parameters['n_nodes']) + '_reps-'
        file = file + str(repetitions) + '_beta-' + str(beta_global) + '_acoes-' + fun(acoes)
        file = file + '_por-(' + fun(parameters["layers_por"])
        file = file + ')_tirar-(' + fun(parameters["layers_tirar"]) + ').pickle'

        exists = os.path.isfile(file)
        if exists:
            with open(file, 'rb') as f:
                count = pickle.load(f)
                f.close()
        else:

            index_list = [i for i in range(1, repetitions + 1)]
            processes = os.cpu_count() - 2
            pool = Pool(processes)
            print('Running with', processes, 'threads')

            count = []
            t1_start = perf_counter()
            result = pool.map(analyze, iterable=index_list, chunksize=None)
            # result =analyze(1)

            t1_stop = perf_counter()
            print("Spent ", t1_stop - t1_start, " seconds")

            count = np.zeros((11, days, repetitions))
            # count[:,:,0] = result
            i = 0
            for it in result:
                count[:, :, i] = it
                i += 1

            with open(file, 'wb') as f:
                pickle.dump(count, f)

    return count

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# [beta_global, parameters] = get_abm_parameters(beta_list=beta_list, which_beta=0)
count0 = simulate_the_abm(parameters = parameters, acoes = acoes, location = location, beta_global = beta_global, repetitions = repetitions)
print(count0)
# count1 = simulate_the_abm(parameters = parameters, layers_0 = layers_0, acoes = acoes, layers_por = layers_por,
#                          layers_tirar = layers_tirar, location = location, beta_global = beta_global[1], repetitions = repetitions)
# count2 = simulate_the_abm(parameters = parameters, layers_0 = layers_0, acoes = acoes, layers_por = layers_por,
#                          layers_tirar = layers_tirar, location = location, beta_global = beta_global[2], repetitions = repetitions)
# count3 = simulate_the_abm(parameters = parameters, layers_0 = layers_0, acoes = acoes, layers_por = layers_por,
#                          layers_tirar = layers_tirar, location = location, beta_global = beta_global[3], repetitions = repetitions)
#
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# errorspace = 2
# plt.figure(0)
# plt.rcParams.update({'font.size': 15})
#
# eb = plt.errorbar(range(0, days), (count0[5] + count0[6])*n, yerr=std_mat[5]*n, lw=2, color='red', label='cases',
#                   errorevery=errorspace)
# eb[-1][0].set_linewidth(1)
#
#
# # plt.ylim(0, 5100000)
# # plt.xlim([0,210])
# plt.xlabel('Days since first confirmed case')
# plt.ylabel('Daily cases')
# ax = plt.gca()
# # ax.yaxis.set_major_formatter(ticker.EngFormatter())
#
# plt.xticks([0, acoes1[0], days],  ("0\n"+primeiro_infectado, str(acoes1[0]) + "\n" + acoes[0], str(days) +"\nDezembro 31"))
#
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()
