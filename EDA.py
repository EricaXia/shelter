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
pet_data.shape
print(pet_data.head())

pet_data['Outcome Type'].groupby(pet_data["Type"]).value_counts(normalize=True) # approx 60% animals get adopted

# df for only euthanized pets
euthanized = pet_data[pet_data['Outcome Type'] == "EUTHANIZE"]

# shows slightly more cats than dogs get euth
euthanized["Type"].value_counts(normalize=True)

# about 62% untreatable get euth
# shockingly, 20% euth were still healthy animals (455 in number)
euthanized["Intake Condition"].value_counts(normalize=True)
euthanized[euthanized["Intake Condition"] == "HEALTHY"].shape


# countplots
plt.title("Outcome Type")
flat_ui=["#2ecc71", "#34495e"]
sns.countplot("Outcome Type", data=pet_data, palette=flat_ui)

plt.title("Outcome Type by Intake Condition")
sns.countplot(x="Outcome Type", hue="Intake Condition", data=pet_data)


# Age distribution
plt.title('Age Distribution')
sns.distplot(pet_data["Age"])

# Age distribution by adopted v euthanized
# Draw Plot
sns.distplot(pet_data.loc[pet_data['Outcome Type'] == 'EUTHANIZE', "Age"], color="dodgerblue", label="EUTHANIZE", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
sns.distplot(pet_data.loc[pet_data['Outcome Type'] == 'ADOPTION', "Age"], color="orange", label="ADOPTION", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
plt.title('Age Distribution by Outcome Type', fontsize=14)
plt.legend()
plt.show()

# adopted v euthanized ; cats v dogs
plt.title('Counts by Pet Type', fontsize=14)
sns.countplot("Outcome Type", hue="Type", data=pet_data)


# adopted v euthanized; different features coefficient values (post modeling)
coef = pd.read_csv("coefficients.csv")
print(coef.shape)
coef.head()
coef['features'] = coef['feature'].str.split('_').str[1]
coef.drop(columns=['feature', 'Unnamed: 0'], inplace=True)

coef.head()

x = coef.loc[:, ['coef']]
coef['coef_z'] = (x - x.mean())/x.std()
coef = coef[(coef["coef_z"] < -0.4) | (coef["coef_z"] > 0.4)]
coef['colors'] = ['red' if x < 0 else 'green' for x in coef['coef_z']]
coef.sort_values('coef_z', inplace=True)
coef.reset_index(inplace=True)


coef.shape

# Draw plot
plt.figure(figsize=(10,8), dpi= 80)
plt.hlines(y=coef.index, xmin=0, xmax=coef.coef_z)
for x, y, tex in zip(coef.coef_z, coef.index, coef.coef_z):
    t = plt.text(x, y, round(tex, 2), 
    horizontalalignment='right' if x < 0 else 'left', verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':14})

# Decorations    
plt.yticks(coef.index, coef.features, fontsize=11)
plt.title('Most Impactful Coefficients on Adoption Outcome', fontdict={'size':14})
plt.grid(linestyle='--', alpha=0.5)
plt.xlim(-6, 3.3)
plt.savefig("eda_last.png", bbox_inches='tight')
plt.show()

