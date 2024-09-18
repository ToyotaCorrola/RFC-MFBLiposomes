from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
  
  data = pd.read_csv("data_file.csv")
# Select features for the classifier (all spectral data)
features = [str(col) for col in range(350, 710, 10)]  

# Prepare the data
X = data[features]
y = (data['num_vesicles'] > 5).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the SVC model
svc_classifier = SVC(kernel='linear', probability=True, random_state=42)
svc_classifier.fit(X_train, y_train)

# Predict probabilities and classes for the test set using SVC
y_pred_svc = svc_classifier.predict(X_test)
y_prob_svc = svc_classifier.predict_proba(X_test)[:, 1]

# Calculate AU-ROC for SVC
roc_auc_svc = roc_auc_score(y_test, y_prob_svc)

# Generate a classification report for SVC
classification_report_svc = classification_report(y_test, y_pred_svc, output_dict=True)

# Prepare the results for the SVC method
svc_results = {
    'Method': 'SVC on All Spectra',
    'Accuracy': classification_report_svc['accuracy'],
    'Precision': classification_report_svc['weighted avg']['precision'],
    'Recall': classification_report_svc['weighted avg']['recall'],
    'AU-ROC': roc_auc_svc
}

svc_results
