"""
Realiza os testes requisitados no projeto
"""
# numpy
import numpy as np
import pandas as pd

# VKFCM
from vkfcm import VKFCM

# metricas
from sklearn.metrics import adjusted_rand_score

# save models
from joblib import dump, load

# Read data
print("Lendo base de dados")
df = pd.read_csv("./data/segmentation.test")
df = df.drop(columns=['REGION-PIXEL-COUNT'])

# create two views
shape_view = df.iloc[:, 1:-9]
rgb_view =  df.iloc[:, -9:]
target = df.iloc[:, 0]

# parametros do cluster
num_clusters = 7
m = 1.6
max_iter = 150
e = 10^(-10)

##
## Função para salvar
##
def fit_model(data, num_clusters, m):
    vkfcm = VKFCM(data, num_clusters, m=m)

    old_cost = vkfcm.cost_function()
    for i in range(150):
        # processo de atualização
        vkfcm.update_centroids()
        vkfcm.update_weights()
        vkfcm.update_membership()

        new_cost = vkfcm.cost_function()
        if np.abs(new_cost - old_cost) < e:
            break
        else:
            old_cost = new_cost

    return vkfcm


def save_model(model, model_name, target):

    centroids = model.centroids
    crisp_partition = model.crisp()
    rand_score = adjusted_rand_score(target, crisp_partition)

    dump(model, './results/' + model_name + '.joblib')
    with open('./results/' + model_name + '.txt', 'w') as f:
        f.write("Rand_score: " + str(rand_score))
        f.write('\n')
        f.write("Peso de cada variavel: " + str(model.weights))
        f.write('\n')
        f.write("Centroids:")
        f.write(str(centroids))
        f.write('\n')
        f.write('Crisp partition:')
        f.write(str(crisp_partition))
        f.write('\n')
###
### Shape View
###
print("Iniciando Shape_View")
best_model = VKFCM(shape_view.values, num_clusters, m=m)
for i in range(20):
    print("Iter: ", i)
    model = fit_model(shape_view.values, num_clusters, m)
    if model.cost_function() < best_model.cost_function():
        best_model = model

# save model
save_model(best_model, 'Shape_view', target)

###
### RGB View
###
"""
print("Iniciando Shape_View")
best_model = VKFCM(shape_view.values, num_clusters, m=m)
for i in range(20):
    print("Iter: ", i)
    model = fit_model(shape_view.values, num_clusters, m)
    if model.cost_function() < best_model.cost_function():
        best_model = model

# save model
save_model(best_model, 'Shape_view', target)
"""
