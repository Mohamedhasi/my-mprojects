import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df=pd.read_csv('C:/Users/Hp/Downloads/titanic.csv')
print(df.head())
#exploring the data
print('----------------------------------------------------------------')

# Summary statistics and information
print(df.info())
print('----------------------------------------------------------------')
print(df.describe())
print('----------------------------------------------------------------')

# Check for missing values
print(df.isnull().sum())
print('----------------------------------------------------------------')

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print(df)
print('----------------------------------------------------------------')

# Convert categorical features to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print(df)
print('----------------------------------------------------------------')

# Drop unnecessary columns
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
print(df)
print('----------------------------------------------------------------')

print('verifying the prprocessing steps')
# Verify the preprocessing steps
print(df.head())
print(df.info())

# Survival Distribution
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution on the Titanic')
plt.xlabel('Difference')
plt.ylabel('Count')
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
plt.show()


#survival distribution by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Distribution by Gender')
plt.xlabel('Difference')
plt.ylabel('Count')
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
plt.legend(title='Sex', labels=['Male', 'Female'])
plt.show()


# Survival by Class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Distribution by Passenger Class')
plt.xlabel('Difference')
plt.ylabel('Count')
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
plt.legend(title='Passenger Class', labels=['1st Class', '2nd Class', '3rd Class'])
plt.show()

# Age Distribution
sns.histplot(df['Age'].dropna(), kde=True, bins=30)
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Separate features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']

# Define numerical and categorical features
numerical_features = ['Age']
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Create preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])# Fill missing values with median

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent
    ('encoder', OneHotEncoder(drop='first'))  # One-hot encode categorical features
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_processed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_processed)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(y_pred[:10])  # Print the first 10 predictions

# Compare predicted labels with true labels
comparison_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})

# Display the comparison dataframe
print(comparison_df.head(10))





