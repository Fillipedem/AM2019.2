"""
Implementação do algoritmo Fuzzy C-Means
"""
import numpy as np


class FCM:

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


    def cost_function(self):
        """
        Calcula a função de custo para a FCM
        """
        ans = 0

        for i in range(len(self.centroids)):

            c = self.centroids[i]

            # calculando norma
            x = self.data - c
            norma = np.sum((x*x), axis=1)

            ans += np.dot(self.membership[:, i]**self.m, norma)


        return ans


    def update_centroids(self):
        """
        Atualizando os centroids
        """

        # for each centroid
        for i in range(len(self.centroids)):

            membership = (self.membership[:, i])**self.m
            self.centroids[i] = np.sum((self.data.T * membership), axis=1) / membership.sum()


    def update_membership(self):
        """
        Atualiza os valores de membership de cada exemplo
        """
        datac = []

        for i in range(len(self.centroids)):
            c = self.centroids[i]

            # calculando norma
            x = self.data - c
            norma = np.sqrt(np.sum((x*x), axis=1))

            datac.append(norma)

        datac = np.array(datac)

        # atualizando o membership
        for i in range(len(self.centroids)):

            soma = 0
            for j in range(len(self.centroids)):
                soma += (datac[i] / datac[j])**(2/(self.m - 1))

            self.membership[:, i] = 1/soma
