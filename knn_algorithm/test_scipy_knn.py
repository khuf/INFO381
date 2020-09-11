#dataset
#http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)

#VIDEO (contains unsuported code that is fixed in this code sample)
#https://www.youtube.com/watch?v=1i0zu9jHN6U
#time:17:48

import numpy as np
import pandas as pd
#from sklearn import preprocessing,cross_validation,neighbors
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors

df = pd.read_csv('knn_sample_data/breast-cancer-wisconsin.data.txt')

#replace question mark with -99999
df.replace('?',-99999,inplace=True)

#delete id column
df.drop(['id'],1,inplace=True)

#remove class row before adding the rest of the row data to X
X = np.array(df.drop(['class'],1))
#Only class row included in the array
y = np.array(df['class'])

#X_train, X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()

#clf.fit arguments: args=['self', 'X', 'y']
clf.fit(X_train,y_train)

#clf.score arguments : args=['self', 'X', 'y', 'sample_weight']
accuracy = clf.score(X_test,y_test)

print("Accuracy:",accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
#add outside array around the array of example measures
example_measures = example_measures.reshape(1,-1)

#clf.predict arguments: args=['self', 'X']
prediction = clf.predict(example_measures)
print("prediction:",prediction)
