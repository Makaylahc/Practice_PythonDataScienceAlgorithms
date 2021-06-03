'''knn.py
K-Nearest Neighbors algorithm for classification
Makaylah cowan
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from palettable import cartocolors


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    
    def __init__(self, num_classes):
        '''KNN constructorsets the number of classes (int: num_classes) this classifier
        will be trained to detect. All other fields initialized to None.'''
        
        self.num_classes = num_classes
        self.exemplars = None
        self.classes = None

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`. '''
        self.exemplars = data 
        self.classes = y
    
    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.'''

        point = []
        for i in range(len(data)):
            pt_1 = data[i]
            distance = (( np.sqrt(np.sum((pt_1 - self.exemplars)**2, axis = 1 )) )) # Eucidean distance
            idx = np.argsort(distance)
            sortedclass = self.classes[idx]
            unique, counts = np.unique(sortedclass[:k], return_counts = True)
            point.append(unique[np.argmax(counts)])

        return np.asarray(point)

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        '''
        return np.sum(y==y_pred) / len(y)

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.'''

        # create sample columns
        samp = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(samp, samp)
       
        # combine two columns
        dat = np.column_stack((x.flatten(), y.flatten())).reshape((x.shape[0]*x.shape[0],2))
        # call predict function to get the predicted classes
        y_pred = self.predict(dat, k).reshape((n_sample_pts, n_sample_pts))

        plt.pcolormesh(x, y, y_pred, cmap = "inferno")
        plt.colorbar()
        plt.show()


    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).
        '''
        result = np.zeros((self.num_classes, self.num_classes))
        
        for actual,predicted in zip(y, y_pred):
            result[int(actual)][int(predicted)] += 1

        return result