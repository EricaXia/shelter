import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path
import sys
from scipy import stats

absFilePath = os.path.join(os.path.dirname('__file__'))
print(absFilePath)
path = os.path.join(absFilePath, r"Data\sonoma_shelter_intake_outcome.csv")
orig_pet_data = pd.read_csv(path)
orig_pet_data.count()
orig_pet_data.shape
# originally 17000 observations

# data frame to experiment with
pet_data = pd.read_csv(path)

# show some information about the data
pet_data.shape
pet_data.head()
pet_data.describe()
pet_data.info()

# drop unwanted columns
pet_data.drop(columns=["Intake Type", "Impound Number", "Kennel Number", "Animal ID", "Intake Date", "Outcome Date", "Days in Shelter", "Name", "Date Of Birth"], axis=1, inplace=True)

# only want cats and dogs
pet_data = pet_data.loc[(pet_data["Type"] == "CAT") | (pet_data["Type"] == "DOG")]

# list null values
print(pet_data.isna().sum())

# remove missing values 
pet_data = pet_data.dropna()


pet_data[['Type', 'Breed', 'Color', 'Sex', 'Size', 'Intake Condition', 'Outcome Type']] = pet_data[['Type', 'Breed', 'Color', 'Sex', 'Size', 'Intake Condition', 'Outcome Type']].astype(str)


# CLEANUP: Age
# Some animals have incorrect age (very large numbers) -> replace outliers w median age

median = pet_data.loc[pet_data['Age'] <= 23, 'Age'].median()
pet_data["Age"] = np.where(pet_data["Age"] > 23, median, pet_data["Age"])


# We will need to standardize the Breed and Color of pets, since there are hundreds of different types. 
# Also rename/combine some types from Sex, Size, etc

# CLEANUP: combine 'unknown' with 'healthy'
pet_data['Intake Condition'] = pet_data['Intake Condition'].str.replace("UNKNOWN", "HEALTHY")


pet_data['Outcome Type'].value_counts()

# CLEANUP: remove rows that were 'return to owner', 'escaped/stolen', then combined everything that's NOT adoption into one 

indexOutcomes = pet_data[(pet_data['Outcome Type'] == "RETURN TO OWNER") | (pet_data['Outcome Type'] == "ESCAPED/STOLEN")| (pet_data['Outcome Type'] == "TRANSFER") ].index
pet_data.drop(indexOutcomes, inplace=True)
pet_data['Outcome Type'] = pet_data['Outcome Type'].str.replace("DIED|DISPOSAL", "EUTHANIZE")


pet_data['Size'].value_counts()
# combine kittens and puppies
pet_data['Size'] = pet_data['Size'].str.replace("KITTN|PUPPY", "KITTN OR PUPPY")


## CLEANUP: Standardize colors 
### split on the '/' into diff columns for primary and secondary colors

pet_data['Color'].value_counts()
new = pet_data['Color'].str.split("/", n = -1, expand = True)
pet_data['First Color'] = new[0]
pet_data['Second Color'] = new[1]
pet_data['Second Color'].fillna(value='', inplace=True)
pet_data['First_Color'] = np.maximum(pet_data['First Color'], pet_data['Second Color'])
pet_data['Second_Color'] = np.minimum(pet_data['First Color'], pet_data['Second Color']) 
# drop unused columns
pet_data.drop(columns=['Color', 'First Color', 'Second Color'], inplace = True)

# reduce number of colors in First_Color category
color_counts1 = pet_data['First_Color'].value_counts()
to_replace = list(color_counts1[color_counts1 < 80].index.values)
def replace_colors(df):
	for i in df.index:
		val = df.loc[i, 'First_Color']
		if val in to_replace:
			df.loc[i, 'First_Color'] = "Other"
# reduce number of colors in Second_Color category
color_counts2 = pet_data['Second_Color'].value_counts()
to_replace2 = list(color_counts2[color_counts2 < 80].index.values)
def replace_colors2(df):
	for i in df.index:
		val = df.loc[i, 'Second_Color']
		if val in to_replace2:
			df.loc[i, 'Second_Color'] = "Other"

replace_colors(pet_data)
replace_colors2(pet_data)


# make all the blank (no color) be in the second color column only
def test_loc(df):
	for i in df.index:
		val = df.loc[i, 'First_Color']
		if val == "":
			df.loc[i, 'First_Color'] = df.loc[i, 'Second_Color']
			df.loc[i, 'Second_Color'] = ""
		elif val == "Other" and df.loc[i, 'Second_Color'] != "":
			df.loc[i, 'First_Color'] = df.loc[i, 'Second_Color']
			df.loc[i, 'Second_Color'] = "Other"
			

test_loc(pet_data)

# check the first 100 values
pet_data.loc[:100, ['First_Color', 'Second_Color']]


# ---------------------------------------------

# CLEANUP: standardize breeds
pet_data['Breed'].nunique()
new1 = pet_data['Breed'].str.split("/", n=-1, expand=True)
pet_data['b1'] = new1[0]
pet_data['b2'] = new1[1]
pet_data['b2'].fillna(value="",inplace=True)
pet_data['First Breed'] = np.maximum(pet_data['b1'], pet_data['b2'])
pet_data['Second Breed'] = np.minimum(pet_data['b1'], pet_data['b2'])
pet_data.drop(columns=['Breed', 'b1', 'b2'], inplace = True)

# reduce number of breeds in First Breed category
breed_counts1 = pet_data['First Breed'].value_counts()
to_replace3 = list(breed_counts1[breed_counts1 < 20].index.values)
# reduce number of breeds in Second Breed category
breed_counts2 = pet_data['Second Breed'].value_counts()
to_replace4 = list(breed_counts2[breed_counts2 < 20].index.values)

def replace_breeds(df):
	for i in df.index:
		val = df.loc[i, 'First Breed']
		val2 = df.loc[i, 'Second Breed']
		if val in to_replace3:
			df.loc[i, 'First Breed'] = "Other"
		if val2 in to_replace4:
			df.loc[i, 'Second Breed'] = "Other"
			
replace_breeds(pet_data)

# clean up breeds columns 
def test_loc2(df):
	for i in df.index:
		val = df.loc[i, 'First Breed']
		val2 = df.loc[i, 'Second Breed']
		if val == "":
			df.loc[i, 'First Breed'] = df.loc[i, 'Second Breed']
			df.loc[i, 'Second Breed'] = ""
		elif val == "MIX" and val2 != "Other":
			df.loc[i, 'First Breed'] = df.loc[i, 'Second Breed']
			df.loc[i, 'Second Breed'] = "MIX"
		elif val == "Other" and val2 == "Other":
			df.loc[i, 'Second Breed'] = ""
		elif val == "Other" and val2 == "MIX":
			df.loc[i, 'First Breed'] = "MIX"
			df.loc[i, 'Second Breed'] = "Other"

test_loc2(pet_data)

# view the first 100 values to check
pet_data.loc[:100, ['First Breed', 'Second Breed']]


# CLEANUP: rename Sex values
pet_data.replace("Spayed", "Female", inplace=True)
pet_data.replace("Neutered", "Male", inplace=True)

print(pet_data.nunique())
# ---------------------------------------
# Encoding categorical variables to numerical dummy variables for the model

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


# Features to train on: Type, Sex, Size, Age, Intake Condition, First_Color, Second_Color, First Breed, Second Breed

numeric_features = ['Age']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
	
categorical_features = ["Type", "Sex", "Size", "Intake Condition", "First_Color", "Second_Color", "First Breed", "Second Breed"]
categorical_transformer = Pipeline(steps = [('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

## -----------------
# Post Preprocessing Final Info:
pet_data.shape
pet_data.nunique()
# Outcome variable: Outcome Type
pet_data.groupby(by=['Type']).count()
# after cleaning, we have 3271 cats and 3335 dogs in the dataset

pet_data.groupby(by=['Outcome Type']).count()
# approximately a 5:2 ratio of adopted:euthanize outcome

# save the cleaned dataset for future use
pet_data.to_csv('cleaned_df.csv')

#--------------------------------

# MODELING

# Append classifier to preprocessing pipeline. Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs', max_iter = 500))])

X = pet_data[["Age", "Type", "Sex", "Size", "Intake Condition", "First_Color", "Second_Color", "First Breed", "Second Breed"]]
y = pet_data['Outcome Type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("model score: %.3f" % clf.score(X_test, y_test))

prob = clf.predict_proba(X_test)[:,1] # gets the predicted probabilities

coef = preprocessing.scale((clf.named_steps['classifier'].coef_).flatten())
coef1 = np.negative(coef)

# get feature names
feature_names = clf.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names()
feature_names = np.insert(feature_names, 0, "Age")

# export coefficients to df and csv file
coef_df = pd.DataFrame({'coef':coef1, 'feature':feature_names})
coef_df = coef_df.sort_values(by='coef')
coef_df.to_csv('coefficients.csv')

# ---------------------------------------------
### Error Analysis and Plots

# convert results to dummies
y_test1 = pd.get_dummies(y_test, drop_first=True)
y_pred1 = pd.DataFrame(y_pred)
y_pred2 = pd.get_dummies(y_pred1, drop_first=True)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Score Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test1, y_pred2)
fpr, tpr, thresholds = roc_curve(y_test1, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('LogReg_ROC')

# Learning Curves
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# defines a function to plot the train/cv learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    #plt.savefig('learning_curve')
    return plt

y_dummies = pd.get_dummies(y, drop_first=True)
title = "Learning Curves (Log Reg)"
cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 0)

plot_learning_curve(clf, title, X, y, cv=cv, n_jobs=4)


# Precision Recall Curve
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test1, y_pred2)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_test1, y_pred2)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: Avg Prec={0:0.2f}'.format(
          average_precision))
plt.savefig('pr_curve')










