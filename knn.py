import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load the iris dataset with pandas
df = pd.read_csv('data/Iris.csv', index_col=0)

# prep the data by mapping species names to discrete values
mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
df.replace({'Species': mapping}, inplace=True)

# split data into feature sets and labels
X = df.drop(['Species'], axis=1)
y = df.Species

# split data into training and testing sets
# NB: train_test_split also randomizes the data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# create a KNN classifier
clf = KNeighborsClassifier(n_neighbors=5)

# train the model
clf.fit(X_train, y_train)

# test the model
print(clf.score(X_test, y_test))