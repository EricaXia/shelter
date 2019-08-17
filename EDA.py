## EDA of Animal Shelter Data

# Questions to explore:
# 1. Why do some animals get adopted? Why do others get euthanized? What sets them apart? Is it due to random chance/luck or certain characteristics?
## 2. How does age play a factor? Intake condit? Colors? Breeds? Type?


## Ideas for plots: histograms, scatterplots w regression line, marginal histogram, heatmap, pairplot, diverging bars (for feature coeffs), countplots, wafflechart  
# # https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python
#    

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# open and preprocess data
pet_data = pd.read_csv('cleaned_df.csv')

pet_data.head()

pet_data['Outcome Type'].value_counts(normalize=True) # approx 60% animals get adopted

# df for only euthanized pets
euthanized = pet_data[pet_data['Outcome Type'] == "EUTHANIZE"]

# shows slightly more cats than dogs get euth
euthanized["Type"].value_counts(normalize=True)

# about 62% untreatable get euth
# shockingly, 20% euth were still healthy animals (455 in number)
euthanized["Intake Condition"].value_counts(normalize=True)

euthanized[euthanized["Intake Condition"] == "HEALTHY"].shape


# Age distribution
sns.distplot(pet_data["Age"])

# Age distribution by adopted v euthanized


# Age distr by type


# adopted v euthanized ; cats v dogs

# adopted v euthanized; colors

