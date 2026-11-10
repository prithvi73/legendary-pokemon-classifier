import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Converting our target matrix values from false and true to 0s and 1s
y = y.astype(int)

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model
clf = RandomForestClassifier()
clf.fit(X_train_scaled, y_train)

# Testing the model
y_pred = clf.predict(X_test_scaled)
score = accuracy_score(y_pred, y_test)
print(score)

# Seeing the model's prediction on individual piece of data
test_subject = X_test_scaled[0:1]
real_answer = y_test.to_numpy()[0].astype(bool)
predicted_answer = clf.predict(test_subject)[0].astype(bool)

print(f"Real Answer : {real_answer}")
print(f"Predicted Answer : {predicted_answer}") 