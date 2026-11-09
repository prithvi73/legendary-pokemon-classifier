import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Loading the data into a dataframe
df = pd.read_csv("Data/Pokemon.csv")
# df.head()

# Dropping megas and primal reversions or any alternate forms
df = df.drop_duplicates(subset=["#"], keep="first")

# Dropping unnecessary columns
df = df.drop(columns=["#", "Name", "Type 1", "Type 2", "Generation"])

# Isolating our features and target matrix
y = df["Legendary"] # Target matrix
df = df.drop(columns=["Legendary"])

X = df.to_numpy() # Features matrix

