# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def generateSpiralData():
    '''
    Generate spiral data for training
    
    N = 300, D = 2, K = 3 will generate dataSets contains 900 2D point.
    900 2D points(D=2) are divided into 3 (K=3) classes, each classes contains
    300 (N=300) points.
    
    dataSet is returned as X, y
    X - pointsArray
    y - classArray
    '''
    np.random.seed(0)
    N = 300 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D))
    y = np.zeros(N*K, dtype='uint8')
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    fig.savefig('spiralData.png')
    return X, y
    
def splitSpiralData(dataSet, labelSet, trainR = 0.6, valR = 0.2):
    '''
    Split data into traingSet, validationSet and testSet
    
    Given data and label array are random shuffled and divided
    trainR=0.6, valR=0.2, dataSet= 100*2 array
    trainData will be 60*2 array, validationdata will be 20*2 array and
    testData will be 20*2 array.
    
    '''
    numSamples = np.shape(dataSet)[0]
    shuffledIndices = np.array(range(numSamples))
    random.shuffle(shuffledIndices)
    
    numTrain = round(numSamples * trainR)
    numVal   = round(numSamples * valR)
    numTest  = numSamples - numTrain - numVal
    
    trainData    = dataSet[shuffledIndices[:numTrain]]
    trainLabels  = labelSet[shuffledIndices[:numTrain]]
    valData      = dataSet[shuffledIndices[numTrain:numTrain + numVal]]
    valLabels    = labelSet[shuffledIndices[numTrain:numTrain + numVal]]
    testData     = dataSet[shuffledIndices[numTrain + numVal:]]
    testLabels   = labelSet[shuffledIndices[numTrain + numVal:]]
    
    return trainData, trainLabels, valData, valLabels, testData, testLabels
 
def initWeightBias(wShape, bShape):
    '''
    Helper function to initilize weight and bias matrix
    
    wShape - Dimension of weight matrix, e.g. (2, 2) 
    bShape - Dimension of bias matrix
    
    '''
    W = 0.01 * np.random.randn(*wShape)
    b = 0.01 * np.zeros(bShape)
    
    return W,b    

def hiddenLayer(X, W, b):        
    # Forward propagation, calculate the value of hidden layer
    #
    # recall in class
    # hidden_layer = ReLU(X * W1 + b1)
    # output_layer = hidden_layer * W2 + b2    
    #
    # implement HINTs 
    # 1. ReLu(x) 
    #     np.maximum(0, x)
    # 2. matrix multiply, A * B
    #     np.dot(A, B)     
    #
    # ===== 2. YOUR CODE STARTs =======   
    hidden_layer = 0.0
    

    
    # ===== 2. YOUR CODE ENDs =======    
    return hidden_layer

def outputLayer(hidden_layer, W, b):
    # Forward propagation, calculate the value of output layer
    #
    # recall in class
    # hidden_layer = ReLU(X * W1 + b1)
    # output_layer = hidden_layer * W2 + b2    
    #
    # ===== 3. YOUR CODE STARTs =======   
    output_layer = 0.0
    
    
    
    
    # ===== 3. YOUR CODE ENDs =======
    return output_layer

def crossEntropy(scores, actLabels):
    '''
    Helper function to calculate the cross-entropy
    
    scores    - the output layer matrix, predicted class label
    actLabels - the actual class label
    
    e.g. There two points P1, P2 need to be classfied, the output
    of our model might look like
    scores    - [[0.02, 0.9, 1.1],      # point 1 score
                 [0.05, 0.7, 0.2]]      # point 2 score
                 
    The probability of each point's class is
    Prob1 - [0.1573, 0.3793, 0.4633]    # 0.3793 = e^0.9 / (e^0.02 + e^0.9 + e^1.1) 
    Prob2 - [0.2453, 0.4698, 0.2849]    

    The actual label of two points
    actLabels -  [1,
                  2]
                  
    Cross-entropy of each point will be
    L1 = -log( 0.3793 ) = 0.9693
    L2 = -log( 0.2849 ) = 1.2556
    
    Output(probs, logprobs) will be 
    probs     = [Prob1,
                 Prob2]
    
    logprobs  = [L1,    
                 L2]

    '''
    numSamples = np.shape(actLabels)[0]
    exp_scores = np.exp(scores)
    probs      = exp_scores/ np.sum(exp_scores, axis=1, keepdims=True)
    logprobs   = -np.log(probs[range(numSamples),trainL])
    
    return probs, logprobs

def dataLoss(Loss):
    # Calculate the data loss from given cross entropy
    #
    # data_loss = (L1 + L2 + ... Ln) / numerOfExamples
    #
    # ===== 4. YOUR CODE STARTs =======       
    data_loss  = 0.0
    
    
    
    # ===== 4. YOUR CODE ENDs =======
    return data_loss

def regLoss(regFactor, W1, W2):
    # Calculate the regularization loss 
    #
    # reg_loss = sum of weights' square
    #
    # ===== 5. YOUR CODE STARTs =======   
    reg_loss = 0.0
    
    
    # ===== 5. YOUR CODE ENDs =======
    return reg_loss

def trainNN(trainD, trainL, D=2, K=3, h=100, step_size=1e-0, reg=1e-3):
    '''
    Train 2 layer neural network
    
    D - input dimensions, e.g. for 2D points, D = 2
    K - output dimensions, e.g. classify 3 classes, K = 2
    h - number of neurons in hidden layer
    step_size - training steps length
    reg - constant for L2 regularization

    Weight and bias of 2 layer neural network is returned
    W1,b1,W2,b2
    '''
    numSamples = np.shape(trainD)[0]
        
    # init Weights and Bias
    # You should initialize W1 W2 with small integer
    # initialize b1 b2 with 0
    #
    # function initWeightBias could be used
    # 
    # W1 - D * h matrix
    # b1 - 1 * h vector
    # W2 - h * K matrix
    # b2 - 1 * K vector
    #    
    # ===== 1. YOUR CODE STARTs =======   
    W1,b1 = initWeightBias((1,1),(1,1))
    W2,b2 = initWeightBias((1,1),(1,1))
    

    # ===== 1. YOUR CODE ENDs =======

    for i in range(10000):
        hidden_layer = hiddenLayer(trainD, W1, b1) 
        output_layer = outputLayer(hidden_layer, W2, b2)
        scores       = output_layer
        
        # Compute the cross entropy and loss
        probs, logprobs = crossEntropy(scores, trainL)              
        data_loss       = dataLoss(logprobs)         
        reg_loss        = regLoss(reg, W1, W2)
        loss            = data_loss + reg_loss
   
        if i%1000 == 0:
            print("iter i %f, loss %f" % (i, loss))
            plotDataAndBoundary(trainD,trainL,W1,b1,W2,b2,fileName='spiral_' + str(i) + '.png')
    
        # compute the gradient on scores, check following article for details
        # http://blog.csdn.net/yc461515457/article/details/51924604
        dscores = probs
        dscores[range(numSamples),trainL] -= 1 
        dscores /= numSamples

        # backward propagation of data_loss
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)

        dhidden = np.dot(dscores, W2.T)
        dhidden[hidden_layer <= 0] = 0 # derivative of reLU
        dW1 = np.dot(trainD.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        # reg_loss
        dW2 += reg* W2
        dW1 += reg* W1

        # update Weight and Bias
        W1 += -step_size * dW1
        b1 += -step_size * db1
        W2 += -step_size * dW2
        b2 += -step_size * db2
    return W1,b1,W2,b2
    
def plotDataAndBoundary(X,y,W1,b1,W2,b2,h=0.02,fileName='spiralAndBoundary.png'):
    '''
    Show data and decision boundary of neural network
    '''
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    fig.savefig(fileName)
    #plt.show()
    
X,y = generateSpiralData()
trainD, trainL, valD, valL, testD, testL = splitSpiralData(X,y)
W1,b1,W2,b2 = trainNN(trainD,trainL)

# test on valDataSet
valLayer1 = hiddenLayer(valD, W1, b1)
valLayer2 = outputLayer(valLayer1, W2, b2)
predicted_class = np.argmax(valLayer2, axis = 1)
print('validation accuracy: %.4f' % (np.mean(predicted_class == valL)))

# plot the resulting classifier
plotDataAndBoundary(valD,valL,W1,b1,W2,b2,fileName='spiralVal.png')

