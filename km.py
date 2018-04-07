import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# load the iris dataset with pandas
df = pd.read_csv('data/Iris.csv', index_col=0)

# store species before they're converted to numbers
species = df.Species.tolist()

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

# create a K Means classifier
clf = KMeans(n_clusters=3)

# train the model
clf.fit(X)

# test the model
print(clf.labels_)
print(species)