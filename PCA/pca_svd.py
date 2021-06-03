'''pca_svd.py
Subclass of PCA_COV that performs PCA using the singular value decomposition (SVD)
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np

import pca_cov


class PCA_SVD(pca_cov.PCA_COV):
    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars` using SVD
        '''
        # Find the data to use for self.A
        self.vars = vars
        if type(self.data[self.vars]) != np.ndarray:
            self.A = self.data[self.vars].to_numpy()
        else:
            self.A = self.data[self.vars]

        self.orgscales = (np.max(self.A, axis = 0) - np.min(self.A, axis =0))
        if normalize == True:
            self.A = (self.A - np.min(self.A, axis =0) ) / self.orgscales

        # Store means
        self.means = np.mean(self.A, axis = 0)

        # Find eigen values/vectors
        U, S, V = np.linalg.svd(self.A - self.means, full_matrices = False)
        self.e_vals = (S * S.T) / (self.A.shape[0]-1)
        #self.e_vals = S**2 / (self.A.shape[0]-1) # another way to get eigen values
        self.e_vecs = V.T

        # Set instance variables
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

        
