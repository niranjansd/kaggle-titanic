# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLogisticRegression
import sklearn
import matplotlib.pyplot as plt
import LoadData
import numpy as np


def fixoutput(submission, testpred):
    #attached predicted output to submission file
    submission.Survived = testpred

    return submission

def randomlr(train_x,train_y,cv_x,test_x,regp,alpha=0.5):
    # Create the random forest object which will include all the parameters
    # for the fit
    randomlr = RandomizedLogisticRegression(C=regp,scaling=alpha,fit_intercept=True,sample_fraction=0.75,n_resampling=200)

    # Fit the training data to the Survived labels and create the decision trees
    randomlr = randomlr.fit(train_x,train_y)

    train_x = randomlr.fit_transform(train_x,train_y)
    cv_x = randomlr.transform(cv_x)
    test_x = randomlr.transform(test_x)

    return train_x,cv_x,test_x

def forestit(train_x,train_y,cv_x,cv_y,test_x,n_est,max_ft=None):
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


#load data and seperate into train and cv
data_x, data_y, test_x, headings, submission = LoadData.loadcleandata()
fraction = 0.66


###MAKE PREDICTIONS FOR SUBMISSION###############
##nummodels = 100
##predictions = np.zeros((test_x.shape[0],nummodels))
##for i in range(nummodels):
##    rseed = np.random.randint(1)
##    train_x,cv_x,train_y,cv_y = sklearn.cross_validation.train_test_split(data_x,data_y,train_size=int(fraction*data_x.shape[0]),random_state=rseed)
##    #select important features using randomized logreg
####    rlrtrain_x,rlrcv_x,rlrtest_x = randomlr(train_x,train_y,cv_x,test_x,regp=1,alpha=0.5)
####    terr,cverr,testpred = forestit(rlrtrain_x,train_y,rlrcv_x,cv_y,rlrtest_x,n_est=50)
##    #train and predict
##    terr,cverr,testpred = forestit(train_x,train_y,cv_x,cv_y,test_x,n_est=100)
##    predictions[:,i] = testpred
##testpred = (np.sum(predictions,axis=1)>(nummodels/2)).astype(int)
###MAKE PREDICTIONS FOR SUBMISSION###############

###VISUALIZING PARAMETERS######################
##n,bins,patched = P.hist([train_x[train_y==1,1],train_x[train_y==0,1]],histtype='bar',stacked=True,color=['blue','green'],label=['Survived','Died'])
##P.xlabel('Gender, 0=female,1=male')
##P.ylabel('Number of passengers, Survived (blue), Died (Green)')
##P.legend(loc='best')
##P.show()
###VISUALIZING PARAMETERS######################

###TESTING PARAMETERS##########################333
rseed = np.random.randint(1)
train_x,cv_x,train_y,cv_y = sklearn.cross_validation.train_test_split(data_x,data_y,train_size=int(fraction*data_x.shape[0]),random_state=rseed)
terrlist = []
cverrlist = []
numiter = train_x.shape[1]
for i in range(numiter):
    terr,cverr,testpred = forestit(train_x,train_y,cv_x,cv_y,test_x,n_est=100,max_ft=i+1)
    terrlist.append(terr)
    cverrlist.append(cverr)

estlist = [i for i in range(numiter)]
plt.plot(estlist,terrlist,'b',estlist,cverrlist,'k')
plt.ylabel('Train Error (Blue), CV Error (Black)')
plt.xlabel('Number of Max features')
#plt.xlabel('Number of Estimators/Trees')
plt.title('Random Forest error curves with 2 features') 
plt.show()
###TESTING PARAMETERS#################################3

submission = fixoutput(submission, testpred)
# (4) Create final submission file
submission.to_csv("submission.csv", index=False)
