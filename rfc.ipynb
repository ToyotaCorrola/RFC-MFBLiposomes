import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

#  Load the data
data=pd.read_csv("data_file.csv") # Replace with your actual data file path

# Prepare the features and target
spectroscopy_features = ["450"]  # Example spectroscopy columns from 400 to 700 nm
X = data[spectroscopy_features]
N_threshold = 1  # Example threshold
y = (data['num_vesicles'] > N_threshold).astype(int)  # Binary target: 1 if num_vesicles > N, else 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]


# Print evaluation metrics
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}')

# Visualize the ROC curve
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_pred_proba)
plt.title('ROC Curve for Random Forest Classifier')
plt.show()

N = 1
# Create a range of thresholds for the '450' column
thresholds = np.linspace(data['450'].min(), data['450'].max(), 100)

# Calculate the probability that more than N vesicles form for each threshold using the new condition (N = 1)
probabilities_n1 = []
for threshold in thresholds:
    count_above_threshold = data[data['450'] >= threshold].shape[0]
    count_above_threshold_and_n_vesicles = data[(data['450'] >= threshold) & (data['num_vesicles'] > N)].shape[0]
    probability_n1 = count_above_threshold_and_n_vesicles / count_above_threshold if count_above_threshold > 0 else 0
    probabilities_n1.append(probability_n1)


# Plot the probability line graph for N > 1 vesicles forming vs. A450 data threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, probabilities_n1, marker='o', color='green')
plt.title('Probability of >' + str(N) + ' Vesicle Forming vs. A450 Data Threshold')
plt.xlabel('A450 Data Threshold')
plt.ylabel('Probability of >1 Vesicle')
plt.ylim(0.0,1.1)
plt.grid(True)
plt.show()
