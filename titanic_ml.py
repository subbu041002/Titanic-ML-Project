import pandas as pd

df = pd.read_csv("train.csv")
print(df.head())

# Drop columns we don't need
df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Fill missing Age values with average age
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Fill missing Embarked values with the most common one
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Convert 'Sex' to numbers: male = 0, female = 1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert 'Embarked' to numbers: S = 0, C = 1, Q = 2
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Optional: Check if any missing values are left
print(df.isnull().sum())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define input features (X) and target label (y)
X = df.drop("Survived", axis=1)  # everything except 'Survived'
y = df["Survived"]               # target column

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
