import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

# # Read In Values from CSV
happy = pd.read_csv("data/totals.csv")
# print(glass.head())
# print(happy.info())

# # Fill missing values
imputer = SimpleImputer(strategy="median")
imputer.fit(happy)
happyFilledArray = imputer.transform(happy)
happyFilled = pd.DataFrame(
    happyFilledArray,
    columns = happy.columns,
    index = happy.index
)
# print(happyFilled.info())

# happyFilled.hist(bins=20)
# plt.show()

train_set, test_set = train_test_split(happyFilled, test_size = 0.2, random_state = 42)
train_labels = train_set["Happiness Score"]
train_data   = train_set.drop(columns = ["Happiness Score"])
test_labels  = test_set["Happiness Score"]
test_data    = test_set.drop(columns = ["Happiness Score"])

corr_matrix = train_set.corr()
corr_relationships = corr_matrix["Happiness Score"].sort_values(ascending = False)
# print(corr_relationships)

# pd.plotting.scatter_matrix(train_set[corr_relationships.index])
# plt.show()

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

train_data_pipelined = num_pipeline.fit_transform(train_data)
test_data_pipelined = num_pipeline.transform(test_data)
# train_data_pipelined = train_data
# test_data_pipelined = test_data

# forset_reg = RandomForestRegressor(n_estimators = 100, max_depth = 10)
# # forset_reg.fit(train_data_pipelined, train_labels)
# forest_scores = cross_val_score(forset_reg, train_data_pipelined, train_labels, scoring = "neg_mean_squared_error", cv = 5)
# print("Error: %0.2f (+/- %0.2f)" % (-forest_scores.mean(), forest_scores.std() * 2))

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

params = {'n_estimators': 311, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
forest_reg = RandomForestRegressor(**params)
# forset_reg_random = RandomizedSearchCV(
#     estimator = forset_reg,
#     param_distributions = random_grid,
#     n_iter = 100,
#     cv = 5,
#     erbose = 1,
#     random_state = 42,
#     n_jobs = -1
# )
# forset_reg_random.fit(train_data_pipelined, train_labels)
# print(forset_reg_random.best_params_)
forest_reg.fit(train_data_pipelined, train_labels)
for name, score in zip(happy.columns[1:],forest_reg.feature_importances_):
    print(name,score)
evaluate(forest_reg, test_data_pipelined, test_labels)

pca = PCA(n_components=2)
pca.fit(train_data_pipelined)
PCAX = pca.transform(train_data_pipelined)
print(pca.explained_variance_ratio_.sum())
plt.scatter(PCAX[:, 0], PCAX[:, 1], c=train_labels)
plt.show()

tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=1000)
tsne_results = tsne.fit_transform(train_data_pipelined)
plt.scatter(PCAX[:, 0], PCAX[:, 1], c=train_labels)
plt.show()
