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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

st.title("Advance Machine Learning - Task 1")
st.subheader("Iris Dataset: Linear Regression vs RNN Classifier")

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("Iris.csv")  # Iris.csv must be in the same folder as this file

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

# ── RNN ────────────────────────────────────────────────────────────────────────
st.write("---")
st.write("## 2. RNN Classifier")

X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn  = X_test.reshape((X_test.shape[0],  1, X_test.shape[1]))

model = Sequential([
    SimpleRNN(32, activation='tanh', input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

with st.spinner("Training RNN... please wait"):
    history = model.fit(
        X_train_rnn, y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_test_rnn, y_test),
        verbose=0
    )

y_pred_rnn       = model.predict(X_test_rnn)
y_pred_rnn_class = np.argmax(y_pred_rnn, axis=1)

acc_rnn = accuracy_score(y_test, y_pred_rnn_class)
st.write(f"**Accuracy:** {acc_rnn:.4f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred_rnn_class))

# ── Plots ──────────────────────────────────────────────────────────────────────
st.write("---")
st.write("## Visualisations")

# 1. RNN Loss Curve
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_title("RNN Loss (Iris)")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# 2. Confusion Matrix
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred_rnn_class)
im = ax.imshow(cm, cmap='Blues')
ax.set_title("Confusion Matrix (RNN)")
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
fig = pd.DataFrame(X_scaled, columns=feature_cols).hist(figsize=(10, 8))
plt.suptitle("Feature Distributions")
st.pyplot(plt.gcf())
plt.close('all')

# 6. Pairplot
df_plot = df.copy()
df_plot['Species'] = le.inverse_transform(df_plot['Species'])
fig = sns.pairplot(df_plot, hue='Species')
st.pyplot(fig.fig)
plt.close('all')

# 7. Actual vs Predicted (first 20)
fig, ax = plt.subplots()
ax.plot(y_test.values[:20], label='Actual')
ax.plot(y_pred_rnn_class[:20], label='Predicted')
ax.set_title("Actual vs Predicted (First 20)")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# 8. Model Comparison
fig, ax = plt.subplots()
ax.bar(["Linear Regression", "RNN"], [acc_lr, acc_rnn], color=['steelblue', 'coral'])
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
for i, v in enumerate([acc_lr, acc_rnn]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
st.pyplot(fig)
plt.close(fig)