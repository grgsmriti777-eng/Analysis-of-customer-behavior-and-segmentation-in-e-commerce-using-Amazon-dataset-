import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, silhouette_score
)

# =========================
# For Google Colab: Upload file manually
# =========================
from google.colab import files
uploaded = files.upload()  # After running, upload your CSV manually

# Load uploaded file - update the file name here to your exact CSV filename
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# =========================
# Inspect and map target
# =========================
print("Unique values in Purchase_Frequency:\n", df['Purchase_Frequency'].value_counts())

freq_map = {
    'Less than once a month': 0,
    'Once a month': 1,
    'Few times a month': 2,
    'Once a week': 3,
    'Multiple times a week': 4
}

df['Purchase_Frequency_Num'] = df['Purchase_Frequency'].map(freq_map)
print("Nulls in target after mapping:", df['Purchase_Frequency_Num'].isnull().sum())

# Drop rows with null target values
df = df.dropna(subset=['Purchase_Frequency_Num'])
df['Purchase_Frequency_Num'] = df['Purchase_Frequency_Num'].astype(int)

# =========================
# Feature selection (adjust as needed)
# =========================
feature_cols = [
    'age', 'Gender', 'Personalized_Recommendation_Frequency',
    'Browsing_Frequency', 'Product_Search_Method', 'Search_Result_Exploration',
    'Customer_Reviews_Importance', 'Add_to_Cart_Browsing', 'Cart_Completion_Frequency'
]

X = df[feature_cols].copy()
y = df['Purchase_Frequency_Num']

# =========================
# Handle missing values
# =========================
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode_val = X[col].mode(dropna=True)
    fill_val = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
    X[col] = X[col].fillna(fill_val)

# =========================
# Encode categorical features
# =========================
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# =========================
# Scale features
# =========================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# PCA (5 components)
# =========================
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio by PCA components: {pca.explained_variance_ratio_}")

# =========================
# Diagnostics: Silhouette & Elbow
# =========================
K_range = range(2, 10)

# Silhouette scores
sil_scores = []
# Elbow (WCSS/Inertia)
inertia_scores = []

for k in K_range:
    kmeans_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_tmp = kmeans_tmp.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels_tmp)
    sil_scores.append(sil)
    inertia_scores.append(kmeans_tmp.inertia_)

best_k_found = K_range[np.argmax(sil_scores)]
print("Best K based on silhouette score (diagnostic):", best_k_found)

# ===== Final KMeans fixed to K=4 =====
FINAL_K = 4

# Plot Silhouette
plt.figure()
plt.plot(list(K_range), sil_scores, marker='o')
plt.title('Silhouette Scores for KMeans Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.axvline(FINAL_K, linestyle='--', linewidth=1, label=f'Chosen K={FINAL_K}')
plt.legend()
plt.show()

# Plot Elbow (Inertia)
plt.figure()
plt.plot(list(K_range), inertia_scores, marker='o')
plt.title('Elbow Method (WCSS/Inertia)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.axvline(FINAL_K, linestyle='--', linewidth=1, label=f'Chosen K={FINAL_K}')
plt.legend()
plt.show()

# =========================
# Final KMeans fit with K=4
# =========================
kmeans_final = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_pca)
df['Cluster'] = cluster_labels
centroids_pca = kmeans_final.cluster_centers_  # centroids in PCA space

# =========================
# Binary classification target (purchase frequency > 1)
# =========================
y_binary = (y > 1).astype(int)

# =========================
# Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.3, random_state=42
)

# =========================
# Logistic Regression with GridSearchCV
# =========================
lr = LogisticRegression(max_iter=500, random_state=42)
lr_params = {'C': [0.01, 0.1, 1, 10]}
lr_gs = GridSearchCV(lr, lr_params, cv=5, scoring='f1')
lr_gs.fit(X_train, y_train)
best_lr = lr_gs.best_estimator_

# =========================
# Decision Tree with GridSearchCV
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt_params = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
dt_gs = GridSearchCV(dt, dt_params, cv=5, scoring='f1')
dt_gs.fit(X_train, y_train)
best_dt = dt_gs.best_estimator_

# =========================
# Predictions and probabilities
# =========================
y_pred_lr = best_lr.predict(X_test)
y_prob_lr = best_lr.predict_proba(X_test)[:, 1]

y_pred_dt = best_dt.predict(X_test)
y_prob_dt = best_dt.predict_proba(X_test)[:, 1]

# =========================
# Evaluation
# =========================
def evaluate_model(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }

metrics_lr = evaluate_model(y_test, y_pred_lr, y_prob_lr)
metrics_dt = evaluate_model(y_test, y_pred_dt, y_prob_dt)

performance_df = pd.DataFrame([metrics_lr, metrics_dt], index=['Logistic Regression', 'Decision Tree'])
print(performance_df)

# =========================
# Confusion matrices
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(cm_lr)
disp_lr.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Logistic Regression Confusion Matrix')

cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(cm_dt)
disp_dt.plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title('Decision Tree Confusion Matrix')

plt.tight_layout()
plt.show()

# =========================
# Feature importance (Decision Tree)
# =========================
feat_importances = pd.Series(best_dt.feature_importances_, index=feature_cols)
top_feats = feat_importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_feats.values, y=top_feats.index)
plt.title('Top 15 Feature Importances - Decision Tree')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# =========================
# KMeans clusters visualization (exactly 4 colors + 4 centroid dots)
# =========================
palette_4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

plt.figure(figsize=(12, 5))

# PCA 0 & 1
plt.subplot(1, 2, 1)
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1],
    hue=cluster_labels,
    palette=palette_4,
    legend='full',
    s=25
)
# Centroids as big unfilled dots
plt.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1],
    s=250, facecolors='none', edgecolors='black', linewidths=2, marker='o', label='Centroid'
)
plt.title('KMeans (K=4): PCA Components 0 & 1')
plt.xlabel('PCA 0')
plt.ylabel('PCA 1')

# PCA 2 & 3
plt.subplot(1, 2, 2)
sns.scatterplot(
    x=X_pca[:, 2], y=X_pca[:, 3],
    hue=cluster_labels,
    palette=palette_4,
    legend='full',
    s=25
)
plt.scatter(
    centroids_pca[:, 2], centroids_pca[:, 3],
    s=250, facecolors='none', edgecolors='black', linewidths=2, marker='o', label='Centroid'
)
plt.title('KMeans (K=4): PCA Components 2 & 3')
plt.xlabel('PCA 2')
plt.ylabel('PCA 3')

# Single legend (clusters + centroid) without duplicates
handles, labels = plt.gca().get_legend_handles_labels()
unique = list(dict(zip(labels, handles)).items())
plt.legend([h for h in [u[1] for u in unique]], [l for l, _ in unique], bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()