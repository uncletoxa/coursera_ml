#!/usr/bin/env python
# coding: utf-8
"""
author -- ToxaZ

Couresera Machine Learning Introduction 2nd week assignement
1 - https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/u08ys/priedobrabotka-dannykh-v-pandas
2 - https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/DVbEI/vazhnost-priznakov
"""
import pandas as pd
import numpy as np
from utils import write_submission
from sklearn.tree import DecisionTreeClassifier


def pandas_assignment():
    data = pd.read_csv('data/titanic.csv', index_col='PassengerId')

    subm11 = data['Sex'].value_counts()
    write_submission(
        ' '.join([str(x) for x in subm11]),
        '11')

    write_submission(
        int(data['Survived'].value_counts(normalize=True).to_dict()[1]*100),
        '12')

    write_submission(
        int(data['Pclass'].value_counts(normalize=True).to_dict()[1]*100),
        '13')

    subm14 = []
    subm14.append(round(float(data['Age'].mean()), 1))
    subm14.append(int(data['Age'].median()))
    write_submission(
        ' '.join([str(x) for x in subm14]),
        '14')

    write_submission(
        round(data.corr('pearson')['SibSp']['Parch'], 2),
        '15')

    write_submission(
        data['Name'].str.extract('(Miss\. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1]
        .value_counts().head(n=1).to_string()
        .split(' ', 1)[0],
        '16')


def trees_assignment():
    data = pd.read_csv('data/titanic.csv', index_col='PassengerId')
    data21 = data.dropna(subset=['Pclass', 'Fare', 'Age', 'Survived', 'Sex'])
    pd.options.mode.chained_assignment = None  # suppress false positive warn
    data21['Sex'] = data21['Sex'].map({'female': 0, 'male': 1})
    pd.options.mode.chained_assignment = 'warn'  # turning warn back
    feature_names = ['Pclass', 'Fare', 'Age', 'Sex']
    X = data21[feature_names]
    y = data21[['Survived']]

    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(X, y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    subm21 = [feature_names[f] for f in indices][:2]

    write_submission([x for x in subm21], '21')


def main():
    pandas_assignment()
    trees_assignment()


if __name__ == '__main__':
    main()
