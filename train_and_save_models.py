import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv("titanic.csv")

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Label encode
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# Split features
X = df.drop(['Survived', 'PassengerId'], axis=1)
y = df['Survived']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
models = {
    "model_logistic_regression.pkl": LogisticRegression(),
    "model_decision_tree.pkl": DecisionTreeClassifier(),
    "model_random_forest.pkl": RandomForestClassifier(),
    "model_naive_bayes.pkl": GaussianNB()
}

for filename, clf in models.items():
    clf.fit(X_scaled, y)
    with open(f"{filename}", "wb") as f:
        pickle.dump(clf, f)

# Save encoders and scaler
pickle.dump(le_sex, open("le_sex.pkl", "wb"))
pickle.dump(le_embarked, open("le_embarked.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ All models and encoders saved successfully.")
