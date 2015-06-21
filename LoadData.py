import numpy as np
import pandas as pd
 
def loadcleandata():
    #Print you can execute arbitrary python code
    train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
    test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )
    all_data = train.append(test)

    print(train.shape,'training samples')
    print(test.shape,'test samples')

    # (2) Create the submission file with passengerIDs from the test file with predicted output
    submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})

    # Prepare the data
    # set up gender
    train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    all_data['Gender'] = all_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # fix missing ages with median age.. this might not be the best strategy
    # find median ages
    median_ages = np.zeros((2,3))
    numagedata = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            numagedata[i,j] = all_data[(all_data['Gender'] == i) & (all_data['Pclass'] == j+1)]['Age'].dropna().count()
            median_ages[i,j] = all_data[(all_data['Gender'] == i) & (all_data['Pclass'] == j+1)]['Age'].dropna().median()
    print(median_ages, 'is the median age among',numagedata, 'number of [female;male]*[Pclass1,2,3]')
    # create ageFill column
    train['AgeFill'] = train['Age']
    test['AgeFill'] = test['Age']
    # Fill in missing ages
    for i in range(0, 2):
        for j in range(0, 3):
            test.loc[ (test.Age.isnull()) & (test.Gender == i) & (test.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
            train.loc[ (train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

    # Fill in missing fare
    numfaredata = all_data['Fare'].dropna().count()
    median_fare = all_data['Fare'].dropna().median()
    print("Median fare is : " + str(median_fare)+' for '+str(numfaredata)+' passengers.' )
    test.loc[test.Fare.isnull(), 'Fare'] = median_fare

    ### Create FamilySize
    ##train['FamilySize'] = train['SibSp'] + train['Parch']
    ##test['FamilySize'] = test['SibSp'] + train['Parch']

    #drop unused colums
    train = train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
    test = test.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1) 

    train_y = train.values[:,0]
    train_x = train.values[:,1:]
    test_x = test.values

    return train_x, train_y, test_x, submission


def sepcv(data,labels,fraction):
    #takes input numpy arrays and fraction between (0,1), dataset and labels and separates it into train
    #and cv datasets according to fraction.
    
    seed = random.seed()
    trainsize = (int(fraction*data.shape[0]),data.shape[1])
    cvsize = (data.shape[0]-trainsize[0],data.shape[1])
    
    train_x = np.zeros(trainsize)
    cv_x = np.zeros(cvsize)
    train_y = np.zeros(trainsize[0],labels.shape[1])
    cv_y = np.zeros(cvsize[0],labels.shape[1])

    ltrain,lcv,vtrain,vcv = sklearn.cross_validation.train_test_split(duration,label,vector,train_size=200,random_state=10)

        
    return traindurs, cvdurs, trainlabels, cvlabels, trainvec, cvvec


