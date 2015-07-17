# Import the random forest package
import sklearn
from sklearn import cross_validation
import matplotlib.pyplot as plt
import LoadData
import numpy as np
import xgboost as xgb


def fixoutput(submission, testpred):
    #attached predicted output to submission file
    submission.Survived = testpred

    return submission

def boostit(train_x,train_y,cv_x,cv_y,test_x,ntree):
    dtrain = xgb.DMatrix(train_x,label=train_y)
    dcv = xgb.DMatrix(cv_x,label=cv_y)
    dtest = xgb.DMatrix(test_x)

    #Set params
    param = {'bst:max_depth':2,'bst:eta':1,'silent':1,'objective':'binary:logistic','nthread':4}
    plst = param.items()

    #Set validation set
    evallist = [(dcv,'eval'),(dtrain,'train')]
    #Training
    numiter = 2
    bst = xgb.train(plst,dtrain,numiter,evallist)
    
    #test the errors on cv
    tpreds = bst.predict(dtrain,ntree_limit=ntree)>0.5
    cvpreds = bst.predict(dcv,ntree_limit=ntree)>0.5
    labels = dtrain.get_label()
    terr = np.sum(tpreds != labels)/len(labels)
    labels = dcv.get_label()
    cverr = np.sum(cvpreds != labels)/len(labels)    

    # this is prediction
    preds = bst.predict(dtest,ntree_limit=ntree)>0.5

    return terr,cverr,preds



    

#load data and seperate into train and cv
data_x, data_y, test_x, headings, submission = LoadData.loadcleandata()
fraction = 0.66
#MAKE PREDICTIONS FOR SUBMISSION###############
##nummodels = 100
##predictions = np.zeros((test_x.shape[0],nummodels))
##for i in range(nummodels):
##    rseed = np.random.randint(i+1)
##    train_x,cv_x,train_y,cv_y = cross_validation.train_test_split(data_x,data_y,train_size=int(fraction*data_x.shape[0]),random_state=rseed)
##    terr,cverr,testpred = boostit(train_x,train_y,cv_x,cv_y,test_x)
##    predictions[:,i] = testpred
##testpred = (np.sum(predictions,axis=1)>(nummodels/2)).astype(int)
###MAKE PREDICTIONS FOR SUBMISSION###############    


###TESTING PARAMETERS##########################333
rseed = np.random.randint(7)
train_x,cv_x,train_y,cv_y = cross_validation.train_test_split(data_x,data_y,train_size=int(fraction*data_x.shape[0]),random_state=rseed)
terrlist = []
cverrlist = []
numiter = 50
for i in range(numiter):
    terr,cverr,testpred = boostit(train_x,train_y,cv_x,cv_y,test_x,i)
    terrlist.append(terr)
    cverrlist.append(cverr)

estlist = [i for i in range(numiter)]
plt.plot(estlist,terrlist,'b',estlist,cverrlist,'k')
plt.ylabel('Train Error (Blue), CV Error (Black)')
plt.xlabel('Number of Max features')
#plt.xlabel('Number of Estimators/Trees')
plt.title('Random Forest error curves with 2 features')
plt
plt.show()
###TESTING PARAMETERS#################################3

submission = fixoutput(submission, testpred)
# (4) Create final submission file
submission.to_csv("submission.csv", index=False)

