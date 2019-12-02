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
        cost = 0
        for i in range(len(self.centroids)):
            for k in range(len(self.data)):
                cost += self.U[k, i]*self.distance(self.data[k], self.centroids[i])

        return cost


    def update_centroids(self):
        """
        Atualizando os centroids
        """
        U = self.U**self.m
        for i in range(len(self.centroids)):

            numerador = np.zeros(self.d)
            denominador = np.zeros(self.d)
            for k in range(len(self.data)):
                soma = U[k, i] * self.gaussian_kernel(self.data[k], self.centroids[i])
                numerador += soma*self.data[k]
                denominador += soma


            self.centroids[i] = numerador/denominador


    def update_weights(self):
        """
        Atualiza os pesos de cada atributo
        """
        U = self.U**self.m

        soma = np.zeros((len(self.centroids), self.d))
        for i in range(len(self.centroids)):

            for k in range(len(self.data)):
                soma[i] += U[k, i] * 2*(1 - self.gaussian_kernel(self.data[k], self.centroids[i]))


        numerador = (np.prod(soma.sum(axis=0)))**(1/self.d)
        for p in range(self.d):
            divisor = soma[:, p].sum()
            self.weights[p] = numerador/divisor


    def update_membership(self):
        """
        Atualiza os valores de U de cada exemplo
        """
        values = {}
        for i in range(len(self.centroids)):
            for k in range(len(self.data)):
                values[(k, i)] = self.distance(self.data[k], self.centroids[i])


        for i in range(len(self.centroids)):
            for k in range(len(self.data)):
                update = 0
                for h in range(len(self.centroids)):
                    if values[(k, h)] == 0:
                        continue
                    update += (values[(k, i)]/values[(k, h)])**(1/(self.m - 1))

                if update != 0:
                    self.U[k, i] = 1/update
                #np.divide(1, update[0], self.U[k, i] , where=update!=0)
    #
    def crisp(self):
        target = []
        for i in range(len(self.U)):
            target.append(np.argmax(self.U[i]))

        return target

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



    def distance(self, x, y):
        """
        Calcula a distancia adaptativa global
        """
        kernel = 2*(1 - self.gaussian_kernel(x, y))
        if len(kernel.shape) > 1:
            ans = self.weights[0]*kernel[:, 0]
        else:
            ans = self.weights[0]*kernel[0]

        for i in range(1, self.d):
            if len(kernel.shape) > 1:
                ans += self.weights[i]*kernel[:, i]
            else:
                ans += self.weights[i]*kernel[i]


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

        # somando para 1 cada cluster/inicializar random
        U = np.random.rand(n, self.num_clusters)
        self.U = U/U.sum(axis=1).reshape(-1, 1)

        # Número de atributos
        self.d = d

        # peso de cada variavel - all multiply by one
        self.weights = np.ones((self.d, 1))

        # Calculando os valores antecipadamente para 2sigma**2
        diff = np.linalg.norm(self.data[None, :, :] - self.data[:, None, :], axis=2)
        diff = diff**2

        sigma = [np.mean([np.quantile(diff[i], 0.1),
                        np.quantile(diff[i], 0.9)]) for i in range(self.d)]

        # Sigma term
        self.sigma = np.array(sigma)
