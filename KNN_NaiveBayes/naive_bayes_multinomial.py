'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Makaylah Cowan
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor sets the number of classes (int: num_classes) this classifier
        will be trained to detect. All other fields initialized to None.'''
        
        self.num_classes = num_classes

        self.class_priors = None
        
        self.class_likelihoods = None

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)
        '''
        # Probability that a test example belongs to each class (class predictions)
        N = data.shape[0]
        M = data.shape[1] # number of features (top words)
        prior= []
        col = []
        like = []
        for i in range(self.num_classes):
            Nc = np.sum(y==i) # number of training samples that belong to class c
            prior.append(Nc/N)
            # total count of word w in emails in class c
            # for each class calc 
            # sum of all of words/features where data is associated with given class
            # 6 values for each class
            # features/word count 
            # rows are emails; columns are different words
            Ncw = np.sum(data[y==i], axis = 0)
            like.append( (Ncw + 1) / ( Ncw.sum() + M ))


        self.class_priors = np.asarray(prior)
        self.class_likelihoods = np.asarray(like) # likelihood that data point belongs to class

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.'''

        classes = []
        for i in range(len(data)):
            prost = []
            for c in range(self.num_classes):
                count = np.sum(np.log(  self.class_likelihoods[c, np.nonzero(data[i])]    ))
                prost.append(np.log(self.class_priors[c])  + count)
            classes.append(np.argmax(prost))
        return np.asarray(classes)

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`. '''
        return (y == y_pred).sum() / len(y)

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).'''
        
        result = np.zeros((self.num_classes, self.num_classes))

        for actual,predicted in zip(y, y_pred):
            result[int(actual)][int(predicted)] += 1

        return result

    def fp_fn(self, y, y_pred):
        '''Gets the first false positive and the first false negative
        '''
        result = np.zeros((self.num_classes, self.num_classes))
        index = 0
        fp = -1
        fn = -1

        for actual, predicted in zip(y, y_pred):
            
            if (predicted == 0) & (actual == 1):
                if fp == -1:
                    fp = index

            if (predicted == 1) & (actual == 0):
                if fn == -1:
                    fn = index

            if (fp != -1) and (fn!= -1):
                return fp, fn

            index += 1
