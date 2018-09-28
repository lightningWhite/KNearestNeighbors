# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 2018

@author: Daniel Hornberger
@brief: This file provides a k-nearest neighbors classifier.
It classifies the iris dataset.

"""
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class KNNModel:        
    """
    A model that uses the k-nearest neighbors algorithm to make classification
    predictions.
    """
    def __init__(self, k, data_train, targets_train):
        self.k = k
        self.reference_data = data_train
        self.reference_targets = targets_train
    
    def get_distance(self, params_1, params_2):
        """
        Calculate the Squared Euclidean distance between the two sets of params.
        The Squared Euclidean distance is used to avoid the computationally 
        expensive square root function. As such, this method does not return
        the actual distance between the two sets of values.
        """
        assert (len(params_1) == len(params_2))
        distance = 0
        
        # Add up the squared differences of each value
        for i in range(len(params_1) - 1):
            distance += (params_2[int(i)] - params_1[int(i)])**2
            
        return distance
        
    def get_value(self, item):
        return item[1]
    
    def predict(self, data_test):
        
        predictions = []
        
        progress = 1
        
        # Loop through each test entry
        for entry in data_test:
            print("Processing {} out of {}".format(progress, len(data_test)))
            
            distances = []
        
            for i in range(len(self.reference_data) - 1):
                # Associate each target to the distance between it and the one in question
                distances.append([self.reference_targets[i], self.get_distance(self.reference_data[i], entry)])
        
            # Sort the distance-target pairs from least to greatest distance
            sorted_distances = sorted(distances, key=self.get_value)
            
            closest_targets = []
        
            # Get the k nearest targets
            if self.k <= 1:
                closest_targets.append(sorted_distances[i][0])
            else:
                closest_targets = [sorted_distances[i][0] for i in range(self.k)]
        
            prediction = -1
            occurence_count = 0
        
            # Analyze the nearest neighbors to make a prediction
            for target in closest_targets:
                # Count how many of k neighbors are this type of target
                target_count = closest_targets.count(target)
                
                # If current target has the most occurences, predict it's this target
                if target_count > occurence_count:
                    occurence_count = target_count
                    prediction = target
                # If there is a tie, set the prediction to be the closest target
                elif (target_count == occurence_count) and (target != prediction):
                    prediction = closest_targets[0]
                
            predictions.append(prediction)
            progress += 1
                
        return predictions
 
class KNNClassifier:  
    """
    A classifier that uses k-nearest neighbors to classsify inputs.
    """    
    def __init__(self, k):
        self.k = k
    
    def fit(self, data_train, targets_train):
        model = KNNModel(self.k, data_train, targets_train)
        return model
    
def main(): 
    # Load the iris dataset
    print("Loading the data...")
    dataset = datasets.load_iris()
#    dataset = datasets.load_digits()
#    dataset = datasets.load_breast_cancer()

    # Obtain a normalizing scaler for scaling new data if added later
    std_scaler = preprocessing.StandardScaler().fit(dataset.data)
    
    # Normalize the data
    std_data = preprocessing.scale(dataset.data)
    

    
    # Randomize the dataset and divide the data for testing and training 
    # The following line uses the iris dataset from sklearn
    data_train, data_test, targets_train, targets_test = train_test_split(
            std_data, dataset.target, test_size = 0.30, random_state=42)
    
    # Declare the number of neighbors to use
    k = 9
    
    # Obtain a classifier. Note: comment/uncomment one or the other
    classifier = KNNClassifier(k) # My custom KNN classifier
#    classifier = KNeighborsClassifier(n_neighbors=k) # sklearn's KNN classifier

    # Fit the data
    print("Training...")
    model = classifier.fit(data_train, targets_train)
    
    # Get the predicted targets
    print("Testing...")
    targets_predicted = model.predict(data_test)
    
    # Calculate and display the accuracy
    print("Calculating the accuracy...")
    num_predictions = len(targets_predicted)    
    correct_count = 0
    for i in range(num_predictions):
        if targets_predicted[i] == targets_test[i]:
            correct_count+=1
            
    accuracy = float(correct_count) / float(num_predictions)
    print("Total Accuracy: {:.2f}%".format(accuracy * 100.0))
    

main()