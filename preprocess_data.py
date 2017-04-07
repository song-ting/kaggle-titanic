import pandas as pd
import sklearn.preprocessing as preprocessing


def preprocess_data(path, scale_param=None):
    df = pd.read_csv(path)
    # 1. sex dummies
    sex_dummies = pd.get_dummies(df.Sex, prefix='Sex')

    df = pd.concat([df, sex_dummies], axis=1)
    # 2. cabin dummies
    df.Cabin.loc[df.Cabin.notnull()] = 'notnull'
    df.Cabin.loc[df.Cabin.isnull()] = 'null'

    cabin_dummies = pd.get_dummies(df.Cabin, prefix='Cabin')

    df = pd.concat([df, cabin_dummies], axis=1)
    # 3. pclass dummies
    pclass_dummies = pd.get_dummies(df.Pclass, prefix='Pclass')
    df = pd.concat([df, pclass_dummies], axis=1)
    # 4. embarked dummies
    embarked_dummies = pd.get_dummies(df.Embarked, prefix='Embarked')

    df = pd.concat([df, embarked_dummies], axis=1)
    # 5. add mean age and mean fare
    df.Age.loc[df.Age.isnull()] = df.Age.mean()
    df.Fare.loc[df.Fare.isnull()] = df.Fare.mean()

    # 6. scale age and fare
    if scale_param is None:
        scaler = preprocessing.StandardScaler()
        age_scale_param = scaler.fit(df.Age)
        fare_scale_param = scaler.fit(df.Fare)
    else:
        scaler = scale_param[0]
        age_scale_param = scale_param[1]
        fare_scale_param = scale_param[2]

    df['Age_scale'] = scaler.fit_transform(df.Age, age_scale_param)
    df['Fare_scale'] = scaler.fit_transform(df.Fare, fare_scale_param)

    return df, (scaler, age_scale_param, fare_scale_param)


def get_train_data(path):
    df, scale_param = preprocess_data(path)
    x_df = df.filter(regex='SibSp|Parch|Sex_.*|Cabin_.*|Pclass_.*|Embarked_.*|Age_.*|Fare_.*')
    x = x_df.as_matrix()
    y = df.Survived

    return x, y, scale_param


def get_test_data(path, scale_param):
    df, scale_param = preprocess_data(path, scale_param)
    x_df = df.filter(regex='SibSp|Parch|Sex_.*|Cabin_.*|Pclass_.*|Embarked_.*|Age_.*|Fare_.*')
    x = x_df.as_matrix()

    return x
