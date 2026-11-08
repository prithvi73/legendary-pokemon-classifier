import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Loading the data into a dataframe
df = pd.read_csv("Data/Pokemon.csv")
# df.head()
