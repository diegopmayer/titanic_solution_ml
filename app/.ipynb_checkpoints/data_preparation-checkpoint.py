import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def data_transform(X):
    '''
    Parameter X:
    X is a list data test inputed at form in index.html
    
    Return:
    X, y, Xtest
        X an y: Use to train the model
        Xtest : Use to Prediction
        
    '''

    X = X
    X = pd.DataFrame(X, columns=['PassengerId', 'Pclass', 'Name', 'Sex',
                                 'Age', 'SibSp', 'Parch', 'Ticket',
                                 'Fare', 'Cabin', 'Embarked'], index=[418])
    df_test = pd.read_csv('C:/Users/diego.mayer/Documents/'
                              'developer/Projects/titanic/'
                              'machine_learning/data/test.csv')
    df_test = pd.concat([df_test, X], axis=0, sort=False)
    df_train = pd.read_csv('C:/Users/diego.mayer/Documents/'
                              'developer/Projects/titanic/'
                              'machine_learning/data/train.csv')
    ytrain = df_train['Survived'].copy()
        
    for df in [df_train, df_test]:

        data = pd.DataFrame()
        data['Pclass'] = df['Pclass'].copy()
        data['Sex'] = df.Sex.map({
            'female':1,
            'male':0
        }).copy()
        #---------------------------------------
        data['Age'] = df['Age'].copy()

        med_cl_fem_3 = data.Age[(data.Pclass == 3) & (data.Sex == 1)].median()
        med_cl_fem_2 = data.Age[(data.Pclass == 2) & (data.Sex == 1)].median()
        med_cl_fem_1 = data.Age[(data.Pclass == 1) & (data.Sex == 1)].median()
        med_cl_mal_3 = data.Age[(data.Pclass == 3) & (data.Sex == 0)].median()
        med_cl_mal_2 = data.Age[(data.Pclass == 2) & (data.Sex == 0)].median()
        med_cl_mal_1 = data.Age[(data.Pclass == 1) & (data.Sex == 0)].median()

        data.Age[(data.Pclass == 3) & (data.Sex == 1)] = data.Age[(data.Pclass == 3) & (data.Sex == 1)].fillna(med_cl_fem_3)
        data.Age[(data.Pclass == 2) & (data.Sex == 1)] = data.Age[(data.Pclass == 2) & (data.Sex == 1)].fillna(med_cl_fem_2)
        data.Age[(data.Pclass == 1) & (data.Sex == 1)] = data.Age[(data.Pclass == 1) & (data.Sex == 1)].fillna(med_cl_fem_1)
        data.Age[(data.Pclass == 3) & (data.Sex == 0)] = data.Age[(data.Pclass == 3) & (data.Sex == 0)].fillna(med_cl_mal_3)
        data.Age[(data.Pclass == 2) & (data.Sex == 0)] = data.Age[(data.Pclass == 2) & (data.Sex == 0)].fillna(med_cl_mal_2)
        data.Age[(data.Pclass == 1) & (data.Sex == 0)] = data.Age[(data.Pclass == 1) & (data.Sex == 0)].fillna(med_cl_mal_1)

        #---------------------------------------
        age_0 = data.Age[data.Age <=4].copy()
        age_1 = data.Age[(data.Age >=5) & (data.Age <=16)].copy()
        age_2 = data.Age[(data.Age >=17) & (data.Age <=26)].copy()
        age_3 = data.Age[(data.Age >=27) & (data.Age <=36)].copy()
        age_4 = data.Age[(data.Age >=37) & (data.Age <=41)].copy()
        age_5 = data.Age[(data.Age >=42) & (data.Age <=62)].copy()
        age_6 = data.Age[data.Age >=63].copy()
        age_0.name, age_1.name, age_2.name = 'age_0', 'age_1', 'age_2'
        age_3.name, age_4.name, age_5.name = 'age_3', 'age_4', 'age_5'
        age_6.name = 'age_6'
        data = pd.concat([data, age_0], axis=1).fillna(0)
        data = pd.concat([data, age_1], axis=1).fillna(0)
        data = pd.concat([data, age_2], axis=1).fillna(0)
        data = pd.concat([data, age_3], axis=1).fillna(0)
        data = pd.concat([data, age_4], axis=1).fillna(0)
        data = pd.concat([data, age_5], axis=1).fillna(0)
        data = pd.concat([data, age_6], axis=1).fillna(0)
        data = data.drop(['Age'], axis=1)

        #---------------------------------------

        data['Family_size'] = df.SibSp + df.Parch

        #---------------------------------------

        data['SibSp'] = df['SibSp'].copy()
        data['Parch'] = df['Parch'].copy()

        #---------------------------------------

        outlier = np.mean(df.Fare) + (2 * np.std(df.Fare))
        fare_out = df.Fare[df.Fare >= outlier].copy()
        fare = df.Fare[df.Fare < outlier].copy()
        fare_out.name, fare.name = 'Fare_out', 'Fare'

        data = pd.concat([data, fare_out], axis=1).fillna(0)
        data = pd.concat([data, fare], axis=1).fillna(0)

        #---------------------------------------

        data['Embarked'] = df['Embarked'].fillna('S').copy()
        embark = pd.get_dummies(data['Embarked'], prefix='Embark')
        data = pd.concat([data, embark], axis=1).copy()
        data = data.drop('Embarked', axis=1).copy()

        #---------------------------------------

        data['Name'] = df.Name.copy()
        surname = ['Mr.', 'Mrs', 'Miss', 'Master', 'Rev.',
                   'Dr', 'Col.', 'Sir.', 'Major', 'Don.', 'Capt']
        for rep in surname:
            for name in data['Name'][data.Name.str.contains(rep, regex=False)]:
                if rep in ['Col.', 'Major', 'Capt']:
                    data.replace(name, 'Army', inplace=True)
                elif rep in ['Sir.', 'Don.']:
                    data.replace(name, 'Mr.', inplace=True)
                else:
                    data.replace(name, rep, inplace=True)

        for title in data.Name:
            if title not in ['Mr.', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Army']:
                data.replace(title, 'Miss', inplace=True)

        name_dum = pd.get_dummies(data.Name, prefix='name').copy()
        data = pd.concat([data, name_dum], axis=1)

        data = data.drop('Name', axis=1)

        #---------------------------------------

        cab = df.Cabin.str.extract((r'([A-Za-z])'), ).fillna(0).copy()
        cab_dum = pd.get_dummies(cab, prefix='Deck')
        data = pd.concat([data, cab_dum], axis=1).copy()
        data = data.drop(['Deck_0'], axis=1).copy()

        #---------------------------------------

        data['Deck_T'] = 0

        #---------------------------------------

        surname = df.Name.str.extract(r'(\w+)').copy()
        surname.columns = ['Surname']

        data = pd.concat([data, surname], axis=1).copy()
        mapping = data.Surname.value_counts()
        data.Surname = data.Surname.map(mapping).copy()

        #---------------------------------------
        null_cabin = df['Cabin'][df['Cabin'].isnull()]
        null_cabin.fillna(1, inplace=True)
        null_cabin.name = 'Null_Cabin'
        data = pd.concat([data, null_cabin], axis=1).fillna(0).copy()

        #---------------------------------------
        mapping_ticket = df['Ticket'].value_counts().copy()
        data['Ticket'] = df['Ticket'].copy()
        data['Ticket'] = data['Ticket'].map(mapping_ticket)

        features = ['Pclass', 'Sex', 'age_0', 'age_2', 'age_3', 'age_4', 'age_5',
           'Family_size', 'SibSp', 'Parch', 'Fare_out', 'Fare', 'Embark_C',
           'Embark_S', 'name_Master', 'name_Miss', 'name_Mr.', 'name_Mrs',
           'Deck_B', 'Deck_E', 'Null_Cabin', 'Surname', 'Ticket']

        if df.shape[1] == df_train.shape[1]:
            Xtrain = data[features].copy()
        else:
            Xtest = data[features].copy()
            
    return Xtrain, ytrain, Xtest