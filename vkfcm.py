"""
Implementação do algoritmo Variable Kernel Fuzzy C-Means-K
"""
import numpy as np


class VKFCM:

    ### init
    def __init__(self, data, num_clusters, m=2):

        # check
        if len(data) < num_clusters:
            raise ValueError("Numéro de clusters menor do que exemplos")

        self.data = data
        self.num_clusters = num_clusters
        self.__initialize(num_clusters)
        self.m = m

    ### Interface
    def predict(self, data):
        ans = np.zeros(len(data))

        #
        distance = self.distance(data, self.centroids[0])

        for i in range(len(self.centroids)):
            # calculando norma
            d =  self.distance(data, self.centroids[i])

            bool_idx = d < distance
            distance[bool_idx] = d[bool_idx]
            ans[bool_idx] = i

        return ans

    def cost_function(self):
        """
        Calcula a função de custo para a VKFCM
        """
        pass


    def update_centroids(self):
        """
        Atualizando os centroids
        """

        # for each centroid
        for i in range(len(self.centroids)):

            c = self.centroids[i]
            kernel = self.distance(self.data, c)
            U = (self.U[:, i])**self.m
            # atualizando
            numerador = np.sum(((self.data.T*kernel) * U), axis=1)
            denominador = (U*kernel).sum()
            np.divide(numerador, denominador,  out=self.centroids[i], where=denominador!=0)


    def update_membership(self):
        """
        Atualiza os valores de U de cada exemplo
        """

        for i in range(len(self.centroids)):
            # centroid Kernel
            d1 = self.distance(self.data, self.centroids[i])
            update = 0
            for h in range(len(self.centroids)):

                # calculando kernel
                d2 = self.distance(self.data, self.centroids[h])
                tmp = np.zeros(len(d1))
                np.divide(d1, d2, out=tmp, where=d2!=0)
                update += tmp**(2/(self.m - 1))

            np.divide(1, update, out=self.U[:, i], where=update!=0)


    ###
    ### Class Methods
    ###
    def gaussian_kernel(self, x, y, p=None):
        """
        Kernel Gaussiano
        """
        if p != None:
            return np.exp(-((x-y)**2/self.sigma[p]))
        else:
            return np.exp(-((x-y)**2/self.sigma))


    def update_weights(self):
        """
        Atualiza os pesos de cada atributo
        """
        U = self.U**self.m
        params_values = np.zeros(self.d)
        for i in range(self.d):

            tmp = 0
            for j in range(len(self.centroids)):
                kernel = 2*(1 - self.gaussian_kernel(self.data[:, i], self.centroids[j, i], i))

                # atualizando
                tmp += np.sum(kernel * U[:, j])

            params_values[i] = tmp

        # Atualizando
        for i in range(self.d):
            if params_values[i] < 0.00005:
                continue
            self.weights[i] = (np.prod(params_values)**(1/self.d))/params_values[i]


    def distance(self, x, y):
        """
        Calcula a distancia adaptativa global
        """
        kernel = 2*(1 - self.gaussian_kernel(x, y))
        ans = self.weights[0]*kernel[:, 0]
        for i in range(1, self.d):
            ans += self.weights[i]*kernel[:, i]

        return ans


    def __initialize(self, num_clusters):
        """
        Inicializa os centroids + membership U
        """
        # genereta random centroids with range [0, 1]
        initial_clusters_index = np.random.randint(low=0, high=len(self.data),
                                                    size=num_clusters)

        self.centroids = self.data[initial_clusters_index] + 1.5

        ### Membership
        n, d = self.data.shape

        # somando para 1 cada cluster
        self.U = np.ones((n, d)) * (1/num_clusters)

        # Número de atributos
        self.d = d

        # peso de cada variavel - all multiply by one
        self.weights = np.ones((self.d, 1))

        # Calculando os valores antecipadamente para 2sigma**2
        diff = np.linalg.norm(self.data[None, :, :] - self.data[:, None, :], axis=2)

        sigma = [np.mean([np.quantile(diff[i], 0.1),
                        np.quantile(diff[i], 0.9)]) for i in range(self.d)]

        # Sigma term
        self.sigma = np.array(sigma)
