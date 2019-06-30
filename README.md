# Titanic-survival-prediction
# This is a basic data science learning project.
# A typical machine learning workflow consists of 7 steps:
# 1 Question or problem definition.
# 2 Acquire training and testing data.
# 3 Wrangle, prepare, cleanse the data.
# 4 Analyze, identify patterns, and explore the data.
# 5 Model, predict and solve the problem.
# 6 Visualize, report, and present the problem solving steps and final solution.
# 7 Supply or submit the results.

# As the problem has ready been defined, the following work will focus on how to find the relationships between features and response, 
then apply them to train the models to find the optimal one to use.

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Read data into dataframe
train_df = pd.read_csv('C:/Users/lance/PycharmProjects/Titanic/train.csv')
test_df = pd.read_csv('C:/Users/lance/PycharmProjects/Titanic/test.csv')
combine = [train_df, test_df]
# Find features in data
print(train_df.columns.values)
# preview the data first and last 5 records to identify feature types and potential errors
print(train_df.head())
print(train_df.tail())

# Find data information and decide how to convert data
print(train_df.info())
print('_'*40)
print(test_df.info())

# summarize all numeric column central tendency, dispersion and shape of a
# dataset’s distribution, excluding NaN values
print(train_df.describe())

# summarize string data type
print(train_df.describe(include=['O']))

# Find if there is any correlation between feature and result
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Use histogram to check if there are any numerical variables like Age bands or ranges dependant patterns
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# Combine multiple features to identify correlations using a single plot
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# Correlate categorical features with survival
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# Correlate categorical and numerical features with survival
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

# Return a tuple representing the dimensionality of the DataFrame
# to show dropping feature results
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
# Extract title related information before dropping Name and ID features
# Use a regular expression to search for a title.  Titles always consist
# of capital and lowercase letters, and end with a period.
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# group sex values by title list based row to correlate with survival
pd.crosstab(train_df['Title'], train_df['Sex'])

# combine rare categories
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Correlate title with survival
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Convert strings to numerical values for model operations
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

# Remove useless features
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

# Show feature correlations in multiple facets
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# Replace NaN and null data with median age based on correlated features sex
# and priority group, ie. median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1,

# Create a 2(sex)x3(Pclass) array of 0s to contain guessed Age values based on Pclass x Gender combinations
guess_ages = np.zeros((2,3))
print(guess_ages)
for dataset in combine:
    # Get all six median age values into array without impact from replacing null values
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    # Replace null values from dropping with Age by corresponding sex&Pclass from the
    # array with stored values
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

# Cut Age to Agebands to correlate Age with survival
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# Create nominal Age bands
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

# Drop Agebands feature
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

# Combine SibSp and Parch features into FamilySize in order to drop them
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Correlate FamilySize with Survived
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Create new feature IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Correlate IsAlone with Survived
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# Drop uncorrelated features

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()
# Create Age*Class feature
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
# Complete missing data with most frequent value
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
# Correlate Embarked with Survived
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',

# Convert Embarked to numeric value                                                                                            ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_df.head()

# Complete missing data with most frequent value in the original dataset
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# Convert continuous numeric value to fare bands
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

# Correlate FareBand with Survive
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

# Convert Fare to numeric values
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# Drop FareBand feature
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

train_df.head(10)
test_df.head(10)


# data visualization


# Take Survived out of X_train and put into label Y_train
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression is a linear model that measures the relationship between the categorical
# dependent variable (feature) and one or more independent variables
# (features) by estimating probabilities

logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, Y_train)

# Predict test results
Y_pred = logreg.predict(X_test)

# Provide the mean accuracy of given data
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Remove columns with zero values
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']

# Calculate coefficient of the features with response (Survived = 1)
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

# Show the results in descending order
coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines, a most widely used clustering non-probabilistic
# binary linear classifier, can also be used for non-linear classification.

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)

# linear_svc is implemented in terms of liblinear rather than libsvm.
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)

# KNN, a non-parametric method used for classification and regression where a sample
# is classified by a majority vote of its neighbors. Better than Logistics Regression but worse than SVM.
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)

# Gaussian naive Bayes assumes continuous values associated with normal distribution.
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)

# Perceptron is a linear predictor function combining a set of weights with the feature vector.
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)

# Stochastic Gradient Descent uses randomly selected (or shuffled) samples to evaluate the gradients.

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)

# Decision_tree is a predictive model which maps features (tree branches) to conclusions about the target value
# (tree leaves). Leaves represent class labels and branches represent conjunctions of features.
# Regression tree takes continuous values. The model confidence score is the highest among models evaluated without ensembeling.
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)

# Random Forests is one of the most popular, constructs a multitude of decision trees (n_estimators=100) at training time
# and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

# Model evaluation

# Make a table to compare models by their accuracy scores
# Pick Random Forest as they correct for decision trees' habit of
# overfitting to their training set.
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)

print(models)





