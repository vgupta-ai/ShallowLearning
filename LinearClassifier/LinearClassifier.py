import numpy as np

numSamples = 300
dimensionOfSamples = 2
numClasses = 3

inputSamples = np.random.randn(numSamples,dimensionOfSamples)
inputClasses = np.random.randint(numClasses, size=numSamples).reshape(numSamples,1)

#model parameters
weightMatrix =0.01 * np.random.randn(dimensionOfSamples,numClasses)
biasMatrix = np.zeros((1,numClasses))

#hyper parameters
step_size = 1e-0
reg = 1e-3

#forward pass
outputMatrix = np.dot(inputSamples,weightMatrix) + biasMatrix

#Exponential used for scores
expOutputMatrix = np.exp(outputMatrix)

#Normalize the scores by the sum of scores of each row to generate probabilities
expOutputMatrixNormalizedDenominator = np.sum(expOutputMatrix, axis=1)
expOutputMatrixNormalizedDenominator = expOutputMatrixNormalizedDenominator.reshape(numSamples,1)
expOutputMatrixNormalized = expOutputMatrix/expOutputMatrixNormalizedDenominator

#Find the correct probability for each class
correctProbabilityMatrix = -np.log(expOutputMatrixNormalized[range(numSamples),inputClasses.T])

data_loss = np.sum(correctProbabilityMatrix)/numSamples
reg_loss = 0.5*reg*np.sum(weightMatrix*weightMatrix)
loss = data_loss + reg_loss

#Back Propagation
dscores = expOutputMatrixNormalized
dscores[range(numSamples),inputClasses.T] -= 1
dscores /= numSamples

dW = np.dot(inputSamples.T, dscores)
db = np.sum(dscores, axis=0)
dW += reg*weightMatrix

#Updation
weightMatrix += -step_size * dW
biasMatrix += -step_size * db