"""
Implementação do algoritmo Kernel Fuzzy C-Means with Kernelization of the Metric
"""
import numpy as np


class KFCM:

    def __init__(self, data, num_clusters, m=2):

        # check
        if len(data) < num_clusters:
            raise ValueError("Numéro de clusters menor do que exemplos")

        self.data = data
        self.num_clusters = num_clusters
        self.centroids = self.__initialize_centroids(num_clusters)
        self.membership = self.__initialize_membership(num_clusters)
        self.m = m


    def __initialize_centroids(self, num_clusters):
        """
        Inicializa os centroids + membership
        """
        # genereta random centroids with range [0, 1]
        n, d = self.data.shape
        centroids = np.random.rand(num_clusters, d)

        # scale [0, 1] to [data.min, data.max]
        data_min, data_max = self.data.min(), self.data.max()
        centroids = np.interp(centroids, (0, 1), (data_min, data_max))

        return centroids


    def __initialize_membership(self, num_clusters):
        """
        Inicializando os valores da membership
        """
        n, d = self.data.shape

        # somando para 1 cada membership
        u = []
        for i in range(len(self.data)):
            x = np.random.dirichlet(np.ones(num_clusters))
            u.append(x)

        return np.array(u)


    def gaussian(self, x, y, sigma):
        x = x - y
        norma = np.sqrt(np.sum((x*x), axis=1))

        ans = np.exp(-(norma**2)/(2*sigma*sigma))

        return ans

    def cost_function(self):
        """
        Calcula a função de custo para a FCM
        """
        ans = 0

        for i in range(len(self.centroids)):

            c = self.centroids[i]

            # calculando norma
            distance = 1 - self.gaussian(self.data, c, 2)
            membership = (self.membership[:, i])**self.m

            ans += np.dot(membership, distance)


        return ans


    def update_centroids(self):
        """
        Atualizando os centroids
        """

        # for each centroid
        for i in range(len(self.centroids)):

            c = self.centroids[i]
            kernel = self.gaussian(self.data, c, 2)
            membership = (self.membership[:, i])**self.m
            # atualizando
            self.centroids[i] = np.sum(((self.data.T*kernel) * membership), axis=1) / (membership*kernel).sum()


    def update_membership(self):
        """
        Atualiza os valores de membership de cada exemplo
        """

        for i in range(len(self.centroids)):
            # centroid Kernel
            centroid_kernel = self.gaussian(self.data, self.centroids[i], 2)
            update = 0
            for h in range(len(self.centroids)):

                # calculando kernel
                kernel = self.gaussian(self.data, self.centroids[h], 2)
                update += ((1 - centroid_kernel)/(1 - kernel))**(2/(self.m - 1))

            self.membership[:, i] = 1 / update




### Teste
from sklearn.datasets import load_iris

iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target

cluster = KFCM(X, 3)
