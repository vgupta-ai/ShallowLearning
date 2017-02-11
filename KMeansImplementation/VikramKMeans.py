import random
import numpy as np
class VikramKMeans:

    numberOfClusters = 1
    finalCentroids = None

    def __init__(self,numberOfClusters,numIterations=float('inf')):
        self.numberOfClusters = numberOfClusters
        self.numIterations = numIterations

    def fit(self,X):
        newCentroids = self.__getRandomCentroids(X,self.numberOfClusters)
        prevCentroids = np.zeros(newCentroids.shape)
        numIter = 0;
        while(self.__shouldStop(prevCentroids,newCentroids)!=True and numIter < self.numIterations):
            prevCentroids = newCentroids
            newCentroids = self.__updateCentroids(newCentroids,X)
            numIter = numIter + 1
            print("New intermediate centroid....")
            print(newCentroids)
        finalCentroids = newCentroids
        return finalCentroids

    def predict(self,X):
        return self

    def __euclideanDistance(self,dataPoints,centroid):
        distance = np.sqrt(((dataPoints-centroid)**2).sum(axis=1))
        return distance

    #Private Functions
    def __updateCentroids(self,centroids,X):
        #given the current centroids, find out the centroid to which each of the point is closest to
        currentLabelsForPoints = self.__predict(X,centroids)
        #Now updating the centroid value depending upon the points belonging to it
        newCentroids = np.zeros(centroids.shape)
        for i in xrange(self.numberOfClusters):
            # currentLabelsForPoints is having the centroids to which each of the point is closest to, now we need to sum those to update the
            # centroid value. From X, we need to find all those X which belong to each of the clusters
            indexsOfDataBelongingToThisCluster = np.where( currentLabelsForPoints == i )[0]
            meanOfThisCluster = np.mean(X[indexsOfDataBelongingToThisCluster,:],axis=0)
            newCentroids[i,:] = meanOfThisCluster
        return newCentroids

    #Find the centroid label to which each of the points are closest to
    def __predict(self,X,centroids):
        distanceFromCentroidMatrix = np.zeros((X.shape[0], self.numberOfClusters))
        for i in xrange(self.numberOfClusters):
            distanceFromCentroidMatrix[:,i] = self.__euclideanDistance(X,centroids[i,:].reshape((1,X.shape[1])))
        labelsForPoints = self._findIndexOfTheMinValueInRow(distanceFromCentroidMatrix)
        return labelsForPoints

    def _findIndexOfTheMinValueInRow(self,X):
        #return np.unravel_index(np.argmin(X,axis=1),X.shape)
        return np.argmin(X, axis=1)

    def __getRandomCentroids(self,X,numberOfClusters):
        numOfSamples = X.shape[0]
        randomIndexes = np.random.randint(X.shape[0], size=numberOfClusters)
        return X[randomIndexes]

    def __shouldStop(self,prevCentroids,newCentroids):
        if(np.array_equal(prevCentroids, newCentroids)):
            return True
        else:
            return False