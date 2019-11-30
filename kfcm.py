"""
Implementação do algoritmo Kernel Fuzzy C-Means with Kernelization of the Metric
"""
import numpy as np


class KFCM:

    ### init
    def __init__(self, data, num_clusters, m=2):

        # check
        if len(data) < num_clusters:
            raise ValueError("Numéro de clusters menor do que exemplos")

        self.data = data
        self.num_clusters = num_clusters
        self.centroids = self.__initialize_centroids(num_clusters)
        self.membership = self.__initialize_membership(num_clusters)
        self.sigma = self.__initialize_sigma()
        self.m = m

    ### Interface
    def predict(self, data):
        ans = np.zeros(len(data))

        #
        distance = (1 - self.gaussian(data, self.centroids[0]))

        for i in range(len(self.centroids)):
            # calculando norma
            d =  (1 - self.gaussian(data, self.centroids[i]))

            bool_idx = d < distance
            distance[bool_idx] = d[bool_idx]
            ans[bool_idx] = i

        return ans


    def gaussian(self, x, y):
        """
        Kernel Gaussiano
        """
        x = x - y
        norma = np.sqrt(np.sum((x*x), axis=1))

        ans = np.exp(-(norma**2)/self.sigma)

        return ans


    def cost_function(self):
        """
        Calcula a função de custo para a KFCM
        """
        ans = 0

        for i in range(len(self.centroids)):

            c = self.centroids[i]

            # calculando norma
            distance = 1 - self.gaussian(self.data, c)
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
            kernel = self.gaussian(self.data, c)
            membership = (self.membership[:, i])**self.m
            # atualizando
            self.centroids[i] = np.sum(((self.data.T*kernel) * membership), axis=1) / (membership*kernel).sum()


    def update_membership(self):
        """
        Atualiza os valores de membership de cada exemplo
        """

        for i in range(len(self.centroids)):
            # centroid Kernel
            centroid_kernel = self.gaussian(self.data, self.centroids[i])
            update = 0
            for h in range(len(self.centroids)):

                # calculando kernel
                kernel = self.gaussian(self.data, self.centroids[h])
                update += ((1 - centroid_kernel)/(1 - kernel))**(2/(self.m - 1))

            self.membership[:, i] = 1 / update




    ###
    ### Class Methods
    ###
    def __initialize_centroids(self, num_clusters):
        """
        Inicializa os centroids + membership
        """
        # genereta random centroids with range [0, 1]
        initial_clusters_index = np.random.randint(low=0, high=len(self.data),
                                                    size=num_clusters)


        centroids = self.data[initial_clusters_index]

        return centroids+1


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


    def __initialize_sigma(self):

        norma = np.linalg.norm(self.data[None, :, :] - self.data[:, None, :], axis=2)

        sigma = np.mean([np.quantile(norma, 0.1),
                        np.quantile(norma, 0.9)])

        return sigma
