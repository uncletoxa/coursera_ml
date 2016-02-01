import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data1 = pd.read_csv('titanic.csv', index_col='PassengerId')
data = data1.dropna(subset=['Pclass', 'Fare', 'Age', 'Survived', 'Sex'])
print data.head()
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
x = data[['Pclass', 'Fare', 'Age', 'Sex']]
y = data[['Survived']]
print x
clf = DecisionTreeClassifier(random_state=241)
clf.fit(x, y)
# importances = clf.feature_importances_
print zip(x, clf.feature_importances_)
