# Import the random forest package
from sklearn.linear_model import RandomizedLogisticRegression
import sklearn
import matplotlib.pyplot as plt
import LoadData
import numpy as np


def fixoutput(submission, testpred):
    #attached predicted output to submission file
    submission.Survived = testpred

    return submission

def randomlr(train_x,cv_x,test_x,regp,alpha=0.5):
    # Create the random forest object which will include all the parameters
    # for the fit
    randomlr = RandomizedLogisticRegression(C=regp,scaling=alpha,fit_intercept=True,sample_fraction=0.75,n_resampling=200)

    # Fit the training data to the Survived labels and create the decision trees
    randomlr = randomlr.fit(train_x,train_y)

    train_x = randomlr.fit_transform(train_x)
    cv_x = randomlr.transform(cv_x)
    test_x = randomlr.transform(test_x)

    return train_x,cv_x,test_x

train_x, train_y, test_x, submission = LoadData.loadcleandata()
fraction = 0.66
train_x,cv_x,train_y,cv_y = sklearn.cross_validation.train_test_split(train_x,train_y,train_size=int(fraction*train_x.shape[0]),random_state=10)

terrlist = []
cverrlist = []
for i in range(20):
    terr,cverr,testpred = logreg(train_x,train_y,cv_x,cv_y,test_x,(2**i)*0.0001,alpha=0.5)
    terrlist.append(terr)
    cverrlist.append(cverr)

estlist = [1+10*i for i in range(20)]
plt.plot(estlist,terrlist,'b',estlist,cverrlist,'k')
plt.ylabel('Train Error (Blue), CV Error (Black)')
plt.xlabel('Number of Estimators/Trees')
plt.title('Random Forest error curves with 2 features') 
plt.show()

submission = fixoutput(submission, testpred)
# (4) Create final submission file
submission.to_csv("submission.csv", index=False)
