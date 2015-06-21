# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
import sklearn
import matplotlib.pyplot as plt
import LoadData
import numpy as np


def forestit(train_x,train_y,cv_x,cv_y,test_x,n_est,max_ft='sqrt'):
    # Create the random forest object which will include all the parameters
    # for the fit
    forest = RandomForestClassifier(n_estimators = n_est,max_features=max_ft)

    # Fit the training data to the Survived labels and create the decision trees
    forest = forest.fit(train_x,train_y)
    #find training and cv error
    trainpred = forest.predict(train_x).astype(int)
    cvpred = forest.predict(cv_x).astype(int)
    terr = 1-np.sum(trainpred == train_y)/trainpred.shape[0]
    cverr = 1-np.sum(cvpred == cv_y)/cvpred.shape[0]
    
    # Take the same decision trees and run it on the test data
    output = forest.predict(test_x).astype(int)

    return terr,cverr,output

def fixoutput(submission, testpred):
    #attached predicted output to submission file
    submission.Survived = testpred

    return submission


train_x, train_y, test_x, submission = LoadData.loadcleandata()
fraction = 0.66
train_x,cv_x,train_y,cv_y = sklearn.cross_validation.train_test_split(train_x,train_y,train_size=int(fraction*train_x.shape[0]),random_state=10)

terrlist = []
cverrlist = []
for i in range(20):
    terr,cverr,testpred = forestit(train_x,train_y,cv_x,cv_y,test_x,n_est=1+10*i)
    terrlist.append(terr)
    cverrlist.append(cverr)

estlist = [1+10*i for i in range(20)]
plt.plot(estlist,terrlist,'b',estlist,cverrlist,'k')
plt.ylabel('Train Error (Blue), CV Error (Black)')
plt.xlabel('Number of Estimators/Trees')
plt.title('Random Forest error curves with sqrt features') 
plt.show()

submission = fixoutput(submission, testpred)
# (4) Create final submission file
submission.to_csv("submission.csv", index=False)
