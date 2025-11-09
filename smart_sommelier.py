import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

st.set_page_config(page_title="The Smart Sommelier", page_icon="🍷", layout="wide")

# --- Title ---
st.title("🍷 The Smart Sommelier")
st.caption("AI-Powered Wine Quality Predictor")

st.markdown("""
Welcome to **The Smart Sommelier** — an AI-driven assistant that predicts wine quality  
based on its chemical properties and gives sommelier-style recommendations.  
You can upload your dataset or use the default red wine dataset.
""")

# --- Upload Dataset ---
uploaded_file = st.file_uploader("📂 Upload a Wine Dataset (CSV with 'quality' column)", type=["csv"])
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv"

try:
    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DEFAULT_DATA_URL)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if "quality" not in df.columns:
    st.error("Dataset must include a 'quality' column.")
    st.stop()

st.subheader("👀 Dataset Preview")
st.dataframe(df.head())

# --- Insights Section ---
with st.expander("📊 Data Insights"):
    st.write(df.describe())
    st.write("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    st.pyplot(fig)

# --- Train Regression Model ---
X = df.drop(columns=["quality"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=300, random_state=42)
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col1, col2 = st.columns(2)
col1.metric("Regression R² Score", f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")

# --- Train Classification Model ---
df["quality_label"] = df["quality"].apply(lambda q: "High" if q >= 7 else ("Medium" if q >= 5 else "Low"))
X_cls = df.drop(columns=["quality", "quality_label"])
y_cls = df["quality_label"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

cls_model = RandomForestClassifier(n_estimators=300, random_state=42)
cls_model.fit(Xc_train, yc_train)
cls_acc = accuracy_score(yc_test, cls_model.predict(Xc_test))

st.metric("Classification Accuracy", f"{cls_acc*100:.2f}%")

# --- Sidebar User Input ---
st.sidebar.header("🍇 Enter Wine Characteristics")
def user_input_features():
    inputs = {}
    for feature in X.columns:
        inputs[feature] = st.sidebar.slider(
            feature,
            float(X[feature].min()),
            float(X[feature].max()),
            float(X[feature].mean())
        )
    return pd.DataFrame([inputs])

input_df = user_input_features()

# --- Predict Regression & Classification ---
pred_quality = reg_model.predict(input_df)[0]
pred_label = cls_model.predict(input_df)[0]

st.subheader("🔮 Predicted Wine Quality")
st.success(f"Estimated Quality Score: **{pred_quality:.2f} / 10**")
st.write(f"🏷️ Quality Category: **{pred_label}**")

# --- Dynamic Recommendation Logic ---
if pred_label == "High":
    tip = "🌟 Exceptional! Well-balanced acidity and perfect alcohol level — ready for fine dining."
elif pred_label == "Medium":
    tip = "🍇 Good wine! Try adjusting pH or residual sugar for smoother texture."
else:
    tip = "⚠️ Low quality — consider improving fermentation duration and alcohol balance."
st.info(tip)

# --- Key Influencing Features ---
top_features = pd.Series(reg_model.feature_importances_, index=X.columns).nlargest(3).index.tolist()
st.write(f"🔍 Key factors influencing your prediction: **{', '.join(top_features)}**")

# --- Feature Importance Chart ---
with st.expander("🧠 Feature Importance Visualization"):
    importance = pd.Series(reg_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importance)

# --- Comparison Chart ---
st.write("### Your Wine Profile vs Dataset Average")
comparison = pd.concat([input_df.iloc[0], X.mean()], axis=1)
comparison.columns = ["Your Wine", "Dataset Average"]
st.bar_chart(comparison)

st.markdown("---")
st.markdown("🧠 *Developed by Smart Sommeliers (Us)")