import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv('data_file.csv')

# List of spectral features (350 to 700 nm range)
spectral_features = list(map(str, range(350, 710, 10)))

# A450 feature
a450_feature = ["450"]

# Target: whether > 1 vesicles formed
y = (data['num_vesicles'] > 1).astype(int)

# Split the data for both A450-only and All Spectra models
X_a450_train, X_a450_test, y_train, y_test = train_test_split(data[a450_feature], y, test_size=0.3, random_state=42)
X_spectra_train, X_spectra_test, _, _ = train_test_split(data[spectral_features], y, test_size=0.3, random_state=42)

# Initialize Random Forest classifiers for both cases
rf_a450 = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_spectra = RandomForestClassifier(n_estimators=1000, random_state=42)

# Train the models
rf_a450.fit(X_a450_train, y_train)
rf_spectra.fit(X_spectra_train, y_train)

# Get predicted probabilities
y_pred_proba_a450 = rf_a450.predict_proba(X_a450_test)[:, 1]
y_pred_proba_spectra = rf_spectra.predict_proba(X_spectra_test)[:, 1]

# Calculate ROC curve and AUC for both models
fpr_a450, tpr_a450, thresholds_a450 = roc_curve(y_test, y_pred_proba_a450)
fpr_spectra, tpr_spectra, thresholds_spectra = roc_curve(y_test, y_pred_proba_spectra)

auc_a450 = auc(fpr_a450, tpr_a450)
auc_spectra = auc(fpr_spectra, tpr_spectra)

# Calculate Youden's J point for A450
J_scores_a450 = tpr_a450 - fpr_a450
J_index_a450 = np.argmax(J_scores_a450)
optimal_threshold_a450 = thresholds_a450[J_index_a450]

# Calculate Youden's J point for All Spectra
J_scores_spectra = tpr_spectra - fpr_spectra
J_index_spectra = np.argmax(J_scores_spectra)
optimal_threshold_spectra = thresholds_spectra[J_index_spectra]

# Calculate Threshold Curve
#######################################
# Initialize range of thresholds
thresholds = np.linspace(0, 1, 10000)  # 10000 points for redundancy 

# Store FPR and TPR for each threshold
fpr_values = []
tpr_values = []

# Scan across thresholds
for threshold in thresholds:
    # Apply the threshold
    y_pred_threshold = (X_a450_test[a450_feature] >= threshold).astype(int)
    
    # Calculate FPR, TPR
    fpr, tpr, _ = roc_curve(y_test, y_pred_threshold)
    
    # Append the first (smallest) FPR and TPR, because roc_curve returns arrays
    fpr_values.append(fpr[1])  # First FPR value after 0
    tpr_values.append(tpr[1])  # First TPR value after 0

# Sort FPR/TPR pairs by FPR
sorted_pairs = sorted(zip(fpr_values, tpr_values))

# Extract sorted FPR and TPR
sorted_fpr = [pair[0] for pair in sorted_pairs]
sorted_tpr = [pair[1] for pair in sorted_pairs]

# Calculate Youden's J for the threshold curve (green line)
J_scores_threshold = np.array(sorted_tpr) - np.array(sorted_fpr)
J_index_threshold = np.argmax(J_scores_threshold)

# Get the optimal FPR and TPR for the threshold curve
optimal_fpr_threshold = sorted_fpr[J_index_threshold]
optimal_tpr_threshold = sorted_tpr[J_index_threshold]

# Calculate AUC using the trapezoidal rule (Riemann sum)
auc_value = np.trapz(sorted_tpr, sorted_fpr)
#######################################

# Plot ROC curve for both models and the threshold line
plt.figure(figsize=(6, 6))  # Adjust aspect ratio to square

# Plot ROC for A450
plt.plot(fpr_a450, tpr_a450, linestyle='--', color='blue', label=f'RF Model A450 (AUC={auc_a450:.2f})')
# Plot Youden J point for A450
plt.scatter(fpr_a450[J_index_a450], tpr_a450[J_index_a450], color='blue', marker='o', label=f'Youden J A450={(tpr_a450[J_index_a450]-fpr_a450[J_index_a450]):.2f}')

# Plot ROC for All Spectra
plt.plot(fpr_spectra, tpr_spectra, linestyle='-', color='red', label=f'RF Model All Spectra (AUC={auc_spectra:.2f})')
# Plot Youden J point for All Spectra
plt.scatter(fpr_spectra[J_index_spectra], tpr_spectra[J_index_spectra], color='red', marker='o', label=f'Youden J All Spectra={(tpr_spectra[J_index_spectra]-fpr_spectra[J_index_spectra]):.2f}')

# Plot Threshold Curve
plt.plot(sorted_fpr, sorted_tpr, linestyle='-', color='green', label=f'Threshold Curve (AUC={auc_value:.2f})')

# Plot Youden's J point for the green curve
plt.scatter(optimal_fpr_threshold, optimal_tpr_threshold, color='green', marker='o', label=f'Youden J Threshold={(optimal_tpr_threshold-optimal_fpr_threshold):.2f}')

# Plot random guessing line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Adjust labels and title
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
