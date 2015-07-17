# Import the random forest package
from sklearn.linear_model import RandomizedLogisticRegression
import sklearn
import matplotlib.pyplot as plt
import LoadData
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class NeuralNetwork:

    def __init__(self, layers, activation='sigmoid'):

        #Choose activation func
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random((layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)



    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones to X, the bias unit
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print 'epochs:', k

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':
    pass

##    nn = NeuralNetwork([2,2,1])
##    X = np.array([[0, 0],
##                  [0, 1],
##                  [1, 0],
##                  [1, 1]])
##    y = np.array([0, 1, 1, 0])
##    nn.fit(X, y)
##    for e in X:
##        print(e,nn.predict(e))

data_x, data_y, test_x, headings, submission = LoadData.loadcleandata()
nn = NeuralNetwork([data_x.shape[1],data_x.shape[1],1])
nummodels = 1
predictions = np.zeros((test_x.shape[0],nummodels))
for i in range(nummodels):
    rseed = np.random.randint(1)
    train_x,cv_x,train_y,cv_y = sklearn.cross_validation.train_test_split(data_x,data_y,train_size=int(fraction*data_x.shape[0]),random_state=rseed)
    #train and predict
    nn.fit(data_x,data_y)
    terr = np.sum(train_y==nn.predict(train_x))
    cverr = np.sum(cv_y==nn.predict(cv_x))
    testpred = nn.predict(test_x))
    predictions[:,i] = testpred
testpred = (np.sum(predictions,axis=1)>(nummodels/2)).astype(int)
###MAKE PREDICTIONS FOR SUBMISSION###############










##def fixoutput(submission, testpred):
##    #attached predicted output to submission file
##    submission.Survived = testpred
##
##    return submission
##
##def randomlr(train_x,cv_x,test_x,regp,alpha=0.5):
##    # Create the random forest object which will include all the parameters
##    # for the fit
##    randomlr = RandomizedLogisticRegression(C=regp,scaling=alpha,fit_intercept=True,sample_fraction=0.75,n_resampling=200)
##
##    # Fit the training data to the Survived labels and create the decision trees
##    randomlr = randomlr.fit(train_x,train_y)
##
##    train_x = randomlr.fit_transform(train_x)
##    cv_x = randomlr.transform(cv_x)
##    test_x = randomlr.transform(test_x)
##
##    return train_x,cv_x,test_x
##
##train_x, train_y, test_x, submission = LoadData.loadcleandata()
##fraction = 0.66
##train_x,cv_x,train_y,cv_y = sklearn.cross_validation.train_test_split(train_x,train_y,train_size=int(fraction*train_x.shape[0]),random_state=10)
##
##terrlist = []
##cverrlist = []
##for i in range(20):
##    terr,cverr,testpred = logreg(train_x,train_y,cv_x,cv_y,test_x,(2**i)*0.0001,alpha=0.5)
##    terrlist.append(terr)
##    cverrlist.append(cverr)
##
##estlist = [1+10*i for i in range(20)]
##plt.plot(estlist,terrlist,'b',estlist,cverrlist,'k')
##plt.ylabel('Train Error (Blue), CV Error (Black)')
##plt.xlabel('Number of Estimators/Trees')
##plt.title('Random Forest error curves with 2 features') 
##plt.show()
##
##submission = fixoutput(submission, testpred)
### (4) Create final submission file
##submission.to_csv("submission.csv", index=False)
