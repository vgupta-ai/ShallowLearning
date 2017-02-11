from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from VikramKMeans import *

numSamples = 1500
numDimensions = 20
numOfClusters = 5
blobDataSetX,blobDataSetY = datasets.make_blobs(n_samples=numSamples,n_features=numDimensions,centers=numOfClusters)

# plt.scatter(blobDataSetX[:,0],blobDataSetX[:,1])
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.show()

print("Actual Clusters....")
for i in xrange(numOfClusters):
    clusterPoints = blobDataSetX[np.where(blobDataSetY==i)[0],:]
    classCentroid = np.mean(clusterPoints, axis=0)
    print(classCentroid)


print("Start Clustering...")
vkMeans = VikramKMeans(numOfClusters)
finalCluster = vkMeans.fit(blobDataSetX)
print("Clustering Done...")
print("Final Clusters...")
print(finalCluster)

