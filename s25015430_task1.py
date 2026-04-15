# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

st.title("Advance Machine Learning - Task 1")
st.subheader("Iris Dataset: Linear Regression vs Neural Network (MLP) Classifier")

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("Iris.csv")

st.write("### Dataset Preview")
st.dataframe(df.head())
st.write("**Shape:**", df.shape)

# ── Preprocessing ──────────────────────────────────────────────────────────────
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

X = df.drop('Species', axis=1)
y = df['Species']
feature_cols = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── Linear Regression ──────────────────────────────────────────────────────────
st.write("---")
st.write("## 1. Linear Regression")

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_lr_class = np.clip(np.round(y_pred_lr).astype(int), 0, 2)

acc_lr = accuracy_score(y_test, y_pred_lr_class)
st.write(f"**Accuracy:** {acc_lr:.4f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred_lr_class))

# ── MLP Neural Network (replaces RNN/TensorFlow) ──────────────────────────────
st.write("---")
st.write("## 2. Neural Network Classifier (MLP)")

with st.spinner("Training Neural Network... please wait"):
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='tanh',
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

st.write(f"**Accuracy:** {acc_mlp:.4f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred_mlp))

# ── Plots ──────────────────────────────────────────────────────────────────────
st.write("---")
st.write("## Visualisations")

# 1. Training Loss Curve
fig, ax = plt.subplots()
ax.plot(mlp.loss_curve_, label='Train Loss')
ax.set_title("Neural Network Loss Curve")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# 2. Confusion Matrix
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred_mlp)
im = ax.imshow(cm, cmap='Blues')
ax.set_title("Confusion Matrix (MLP)")
plt.colorbar(im, ax=ax)
for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, cm[i][j], ha='center', va='center', color='black')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
plt.close(fig)

# 3. Class Distribution
fig, ax = plt.subplots()
df['Species'].value_counts().plot(kind='bar', ax=ax)
ax.set_title("Class Distribution")
ax.set_xlabel("Species")
ax.set_ylabel("Count")
st.pyplot(fig)
plt.close(fig)

# 4. Feature Correlation Heatmap
fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(X_scaled, columns=feature_cols).corr(),
            annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Feature Correlation Heatmap")
st.pyplot(fig)
plt.close(fig)

# 5. Feature Distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for i, col in enumerate(feature_cols):
    axes[i // 2][i % 2].hist(X_scaled[:, i], bins=20)
    axes[i // 2][i % 2].set_title(col)
plt.suptitle("Feature Distributions")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# 6. Pairplot
df_plot = df.copy()
df_plot['Species'] = le.inverse_transform(df_plot['Species'])
fig = sns.pairplot(df_plot, hue='Species')
st.pyplot(fig.fig)
plt.close('all')

# 7. Actual vs Predicted (first 20)
fig, ax = plt.subplots()
ax.plot(y_test.values[:20], label='Actual', marker='o')
ax.plot(y_pred_mlp[:20], label='Predicted', marker='x')
ax.set_title("Actual vs Predicted (First 20)")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# 8. Model Comparison
fig, ax = plt.subplots()
bars = ax.bar(["Linear Regression", "Neural Network (MLP)"], [acc_lr, acc_mlp],
              color=['steelblue', 'coral'])
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
for bar, v in zip(bars, [acc_lr, acc_mlp]):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
            f"{v:.3f}", ha='center')
st.pyplot(fig)
plt.close(fig)