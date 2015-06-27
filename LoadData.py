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

    # set up embark port data
    #set the 2 null cases to southampton S which the most common embark port.
    train['Embarked'].fillna('N',inplace = True)
    train['NEmbarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q':2, 'N':0} ).astype(int)
    test['Embarked'].fillna('N',inplace = True)
    test['NEmbarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q':2, 'N':0} ).astype(int)
    
    # fix missing ages/fares with median age/fare.. this might not be the best strategy
    # find median ages/fares
    median_ages = np.zeros((2,3))
    numagedata = np.zeros((2,3))
    median_fares = np.zeros((2,3))
    numfaredata = np.zeros((2,3))
    all_data.Fare = all_data.Fare.map(lambda x: np.nan if x==0 else x)
    for i in range(0, 2):
        for j in range(0, 3):
            numagedata[i,j] = all_data[(all_data['Gender'] == i) & (all_data['Pclass'] == j+1)]['Age'].dropna().count()
            median_ages[i,j] = all_data[(all_data['Gender'] == i) & (all_data['Pclass'] == j+1)]['Age'].dropna().median()
            numfaredata[i,j] = all_data[(all_data['Gender'] == i) & (all_data['Pclass'] == j+1)]['Fare'].dropna().count()
            median_fares[i,j] = all_data[(all_data['Gender'] == i) & (all_data['Pclass'] == j+1)]['Fare'].dropna().median()
#    print(median_ages, 'is the median age among',numagedata, 'number of [female;male]*[Pclass1,2,3]')
#    print(median_fares, 'is the median fare among',numfaredata, 'number of [female;male]*[Pclass1,2,3]')
    # create AgeFill/FareFill columns
    train['AgeFill'] = train['Age']
    test['AgeFill'] = test['Age']
    train['FareFill'] = train['Fare']
    test['FareFill'] = test['Fare']
    # Fill in missing ages/fares
    for i in range(0, 2):
        for j in range(0, 3):
            test.loc[ (test.Age.isnull()) & (test.Gender == i) & (test.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
            train.loc[ (train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
            test.loc[ ((test.Fare.isnull()) | (test['Fare']==0)) & (test.Gender == i) & (test.Pclass == j+1), 'FareFill'] = median_fares[i,j]
            train.loc[ ((train.Fare.isnull()) | (train['Fare']==0)) & (train.Gender == i) & (train.Pclass == j+1), 'FareFill'] = median_fares[i,j]


    ### Create FamilySize
    train['FamilySize'] = (train['SibSp']>0).astype(float) +(train['SibSp']>3).astype(float)
    test['FamilySize'] = (train['SibSp']>0).astype(float) +(train['SibSp']>3).astype(float)

    ### Create NewParch
    train['NewParch'] = (train['Parch']>0).astype(float) +(train['Parch']>2).astype(float)
    test['NewParch'] = (train['Parch']>0).astype(float) +(train['Parch']>2).astype(float)

    ### Create NewAge
    train['NewAge'] = train['AgeFill'] < 0
    test['NewAge'] = test['AgeFill'] < 0
    train['NewAge'] = train['NewAge'] + (train['Gender']*(train['AgeFill']>10)).astype(float) \
                      +(train['Gender']*(train['AgeFill']>25)).astype(float) \
                      +(train['Gender']*(train['AgeFill']>55)).astype(float)
    test['NewAge'] = test['NewAge'] + (test['Gender']*(test['AgeFill']>10)).astype(float) \
                      +(test['Gender']*(test['AgeFill']>25)).astype(float) \
                      +(test['Gender']*(test['AgeFill']>55)).astype(float)
    train['NewAge'] = train['NewAge'] + ((2+~train['Gender'])*(train['AgeFill']>9)).astype(float) \
                      +((2+~train['Gender'])*(train['AgeFill']>13)).astype(float) \
                      +((2+~train['Gender'])*(train['AgeFill']>50)).astype(float)
    test['NewAge'] = test['NewAge'] + ((2+~test['Gender'])*(test['AgeFill']>9)).astype(float) \
                      +((2+~test['Gender'])*(test['AgeFill']>13)).astype(float) \
                      +((2+~test['Gender'])*(test['AgeFill']>50)).astype(float)

##    ### Create genderclass
##    train['GenderClass'] = (train['Parch']>0).astype(float) +(train['Parch']>2).astype(float)
##    test['GenderClass'] = (train['Parch']>0).astype(float) +(train['Parch']>2).astype(float)

    #drop unused colums
    train = train.drop(['PassengerId', 'Name', 'Sex', 'Ticket','Fare', 'Cabin', 'Embarked', 'Age','AgeFill','SibSp','Parch'], axis=1)
    test = test.drop(['PassengerId', 'Name', 'Sex', 'Ticket','Fare', 'Cabin', 'Embarked', 'Age','AgeFill','SibSp','Parch'], axis=1) 

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


