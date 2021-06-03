'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Makaylah Cowan
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):

        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        self.means = None

        self.orgscales = None

    def get_prop_var(self):
        return self.prop_var

    def get_cum_var(self):
        return self.cum_var

    def get_eigenvalues(self):
        return self.e_vals

    def get_eigenvectors(self):
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`
        '''
        return np.cov(data, rowvar= False)

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).
        '''
        # Divide each eigenvalues by sum of eigen values
        self.prop_var = np.ndarray.tolist(e_vals / np.sum(e_vals))
        return self.prop_var

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).
        '''
        self.cum_var = np.ndarray.tolist( np.cumsum(prop_var) )
        return self.cum_var 

        #he Cumulative % column gives the percentage of variance accounted for by the first n components

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`
        '''

        # Find the data to use for self.A
        self.vars = vars
        if type(self.data[self.vars]) != np.ndarray:
            print(type(self.data[self.vars]))
            self.A = self.data[self.vars].to_numpy()
        else:
            self.A = self.data[self.vars]

        self.orgscales = ( np.max(self.A, axis = 0) - np.min(self.A, axis =0 ) )
        if normalize == True:
            self.A = ( self.A - np.min(self.A, axis = 0) ) / self.orgscales

        # Find eigen values/vectors
        self.e_vals, self.e_vecs = np.linalg.eig(self.covariance_matrix(self.A))

        # Set instance variables
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

        # Store means
        self.means = np.mean(self.A)


    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        '''
        # Show specified number of principal components(x) and the corresponding number of amount of explained variance(y)
        if num_pcs_to_keep != None:
            x = range(num_pcs_to_keep)
            y = self.cum_var[:num_pcs_to_keep]
            if num_pcs_to_keep < 40:
                plt.xticks(np.arange(min(x), max(x)+1, 1.0))
            plt.plot(x, y, marker = 'x')

        # Show all principal components(x) and the corresponding amount of explained variance(y)
        else:
            x = range(len(self.cum_var))
            y = self.cum_var
            if len(self.vars) < 40:
                plt.xticks(np.arange(min(x), max(x)+1, 1.0))
            plt.plot(x, y, marker = 'x')

        # Label axis
        plt.xlabel("Principal Components")
        plt.ylabel("Proportion of Variance Explained")
        plt.title("Cumulative Variance Across PCs")


    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)
        '''
        self.A_proj = (self.A - self.means) @ self.e_vecs[:, pcs_to_keep]
        return self.A_proj

    def loading_plot(self):
        '''Create a loading plot of the top 2 PC eigenvectors

        '''
        # Create heatmap with eigenvectors
        fig, ax = plt.subplots()
        im = ax.imshow(np.abs(self.e_vecs), cmap='gray')

        # Label the heatmap
        ax.set_xticks(np.arange(len(self.e_vecs)))
        ax.set_yticks(np.arange(len(self.vars)))
        ax.set_yticklabels(self.vars)
        ax.set_xticklabels( [i for i in range(len(self.e_vecs))] )
        fig.colorbar(im)


    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        '''
        top = []
        for i in range(top_k):
            top.append(i)

        self.pca_project(top) # do PCA

        # Project back
        return self.A_proj @ self.e_vecs[:,:top_k].T + self.means # rotate it back and add the means back on
