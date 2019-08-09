import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# open and preprocess data
pet_data = pd.read_csv('cleaned_df.csv')
numeric_features = ['Age']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_features = ["Type", "Sex", "Size", "Intake Condition", "First_Color", "Second_Color", "First Breed", "Second Breed"]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#--------------------------------

# MODELING
# full prediction pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='newton-cg', max_iter = 500))])

X = pet_data[["Age", "Type", "Sex", "Size", "Intake Condition", "First_Color", "Second_Color", "First Breed", "Second Breed"]]
y = pet_data['Outcome Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# list null values
print(X_train.isna().sum())

clf.fit(X_train, y_train) # fit model
y_pred = clf.predict(X_test)
print("model score: %.3f" % clf.score(X_test, y_test))

prob = clf.predict_proba(X_test)[:,0] # gets predicted probabilities

# -----------------------
#creating and training a model
#serializing our model to a file 
pickle.dump(clf, open("clf_model.pkl","wb"))

# ----------------------
# TODO: write function that takes raw input, converts to proper encoded array of 93 features, feeds it to model, get prediction out

cols = ['Age', 'Type', 'Sex', 'Size', 'Intake Condition', 'First_Color',
       'Second_Color', 'First Breed', 'Second Breed']

def ValuePredictor(to_predict_list):
    to_predict_dict = {k: v for k, v in zip(cols, to_predict_list)}
    to_predict = pd.DataFrame(data=to_predict_dict, columns=cols, index=[0])
    loaded_model = pickle.load(open("clf_model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    prob = loaded_model.predict_proba(to_predict)[:,0]
    return(result[0], prob[0])

example_list = [4.0, 'DOG', 'Male', 'SMALL', 'HEALTHY', 'BLACK', 'Other', 'CHIHUAHUA SH', 'BEAGLE']

results = ValuePredictor(example_list)
'fsfcsd' + '{:.2%}'.format(np.round(results[1], decimals=2))

