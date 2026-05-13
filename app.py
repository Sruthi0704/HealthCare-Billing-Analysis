import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_healthcare_billing.csv")

df = load_data()

# -------------------------------
# TITLE
# -------------------------------
st.title("🏥 Healthcare Billing Analysis & Cost Prediction System")
st.markdown("Analyze healthcare billing patterns and predict treatment costs using machine learning.")

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
st.sidebar.header("🔍 Filters")

condition_filter = st.sidebar.selectbox(
    "Select Medical Condition",
    df["Medical Condition"].unique()
)

filtered_df = df[df["Medical Condition"] == condition_filter]

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(filtered_df.sample(min(20, len(filtered_df))))

# -------------------------------
# KPIs
# -------------------------------
total_revenue = filtered_df['Billing Amount'].sum()
approval_rate = (filtered_df['Claim_Status'] == 'Approved').mean() * 100
avg_stay = filtered_df['Length_of_Stay'].mean()

st.subheader("📌 Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
col2.metric("✅ Approval Rate", f"{approval_rate:.2f}%")
col3.metric("⏳ Avg Stay", f"{avg_stay:.1f} days")

# -------------------------------
# VISUALIZATION STYLE
# -------------------------------
sns.set_style("whitegrid")

# -------------------------------
# ROW 1: BAR + PIE
# -------------------------------
col1, col2 = st.columns(2)

# Revenue by Insurance
with col1:
    st.subheader("🏥 Revenue by Insurance Provider")
    fig, ax = plt.subplots()
    filtered_df.groupby('Insurance Provider')['Billing Amount'].sum().plot(
        kind='bar',
        color='skyblue',
        ax=ax
    )
    plt.xticks(rotation=30)
    st.pyplot(fig)

# Claim Status Pie
with col2:
    st.subheader("📊 Claim Status Distribution")
    fig, ax = plt.subplots()
    filtered_df['Claim_Status'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("pastel"),
        ax=ax
    )
    ax.set_ylabel('')
    st.pyplot(fig)

# -------------------------------
# ROW 2: HISTOGRAM + PIE (NEW)
# -------------------------------
col3, col4 = st.columns(2)

# Length of Stay
with col3:
    st.subheader("📈 Length of Stay Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['Length_of_Stay'], bins=10, color='green', ax=ax)
    st.pyplot(fig)

# Average Billing Pie
with col4:
    st.subheader("💊 Billing Share by Medical Condition")
    fig, ax = plt.subplots()
    avg_cost = filtered_df.groupby('Medical Condition')['Billing Amount'].mean()

    avg_cost.plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("pastel"),
        ax=ax
    )
    ax.set_ylabel('')
    st.pyplot(fig)

# -------------------------------
# ROW 3: HISTOGRAM + SCATTER
# -------------------------------
col5, col6 = st.columns(2)

# Billing Distribution
with col5:
    st.subheader("📊 Billing Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['Billing Amount'], bins=15, kde=True, color='purple', ax=ax)
    st.pyplot(fig)

# Scatter Plot
with col6:
    st.subheader("🔍 Billing vs Length of Stay")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='Length_of_Stay',
        y='Billing Amount',
        hue='Medical Condition',
        data=filtered_df,
        ax=ax
    )
    st.pyplot(fig)

# -------------------------------
# MODEL (CACHED)
# -------------------------------
@st.cache_resource
def train_model(data):
    model_df = data[['Medical Condition', 'Admission Type', 'Length_of_Stay', 'Billing Amount']]
    model_df = pd.get_dummies(model_df, drop_first=True)

    X = model_df.drop('Billing Amount', axis=1)
    y = model_df['Billing Amount']

    model = RandomForestRegressor()
    model.fit(X, y)

    return model, X.columns

model, X_cols = train_model(df)

# -------------------------------
# PREDICTION
# -------------------------------
st.subheader("💰 Treatment Cost Prediction")
st.info("Prediction is based on historical billing data.")

col7, col8, col9 = st.columns(3)

condition_input = col7.selectbox("Disease", df["Medical Condition"].unique())
admission_input = col8.selectbox("Admission Type", df["Admission Type"].unique())
stay_input = col9.slider("Length of Stay", 1, 30, 10)

input_df = pd.DataFrame({
    "Medical Condition": [condition_input],
    "Admission Type": [admission_input],
    "Length_of_Stay": [stay_input]
})

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X_cols, fill_value=0)

if st.button("🔍 Predict Cost"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Treatment Cost: ₹{prediction:,.2f}")

# -------------------------------
# RECOMMENDATIONS
# -------------------------------
st.subheader("📌 Recommendations")

unpaid_rate = (df['Payment_Status'] == 'Unpaid').mean() * 100

if unpaid_rate > 30:
    st.warning("⚠️ High unpaid rate — improve payment collection system")

st.write("✔ Improve claim approval process")
st.write("✔ Reduce high-cost treatments")
st.write("✔ Optimize claim processing time")