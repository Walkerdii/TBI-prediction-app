import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# ==================== Global Config ====================
AGE_BINS = {
    0: "<5", 1: "5-9", 2: "10-14", 3: "15-19",
    4: "20-24", 5: "25-29", 6: "30-34", 7: "35-39",
    8: "40-44", 9: "45-49", 10: "50-54", 11: "55-59",
    12: "60-64", 13: "65-69", 14: "70-74", 15: "75-79",
    16: "80-84", 17: "85-89", 18: "90-94", 19: "95+"
}
SEX_MAP = {0: "Female", 1: "Male"}
DIAGNOSIS_MAP = {
    0: "Head_Injuries",
    1: "Minor_TBI",
    2: "Moderate_Severe_TBI"
}
SHAP_COLOR = "#2E86AB"

# ==================== Data Processing ====================
@st.cache_data
def load_data():
    """Load and process data"""
    try:
        # 从GitHub仓库读取数据
        df = pd.read_excel(
            "final_result.xlsx",  # 确保文件在repo根目录
            engine="openpyxl"
        )

        # Encode categorical variables
        le_age = LabelEncoder()
        df["age"] = le_age.fit_transform(df["age_name"])
        le_sex = LabelEncoder()
        df["sex"] = le_sex.fit_transform(df["sex_name"])
        le_diag = LabelEncoder()
        df["diagnosis"] = le_diag.fit_transform(df["diagnosis"])

        features = df[["log_population", "sex", "age", "year", "diagnosis"]]

        models = {
            "YLDs": XGBModelWrapper().train(features, df["YLDs"]).model,
            "Incidence": XGBModelWrapper().train(features, df["Incidence"]).model,
            "Prevalence": XGBModelWrapper().train(features, df["Prevalence"]).model
        }

        return models, df

    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        st.stop()


class XGBModelWrapper:
    def __init__(self):
        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self


# ==================== Page Setup ====================
st.set_page_config(
    page_title="TBI Burden Predictor",
    layout="wide",
    page_icon="🏥"
)
st.title("Traumatic Brain Injury Burden Prediction System")
st.subheader("XGBoost-based Predictive Model with Explainable AI")
st.caption("Data Source: GBD Database | Developed by: Your Research Team")

# ==================== Sidebar Inputs ====================
with st.sidebar:
    st.header("⚙️ Prediction Parameters")
    age_group = st.selectbox("Age Group", list(AGE_BINS.values()))
    sex = st.radio("Gender", ["Female", "Male"])
    year = st.slider("Year", 1990, 2050, 2023)
    population = st.number_input(
        "Population (Millions)",
        min_value=1,
        value=10,
        help="Actual population = Input value × 1,000,000"
    )
    diagnosis = st.selectbox("Diagnosis Category", list(DIAGNOSIS_MAP.values()))

# ==================== Encode Inputs ====================
age_code = [k for k, v in AGE_BINS.items() if v == age_group][0]
sex_code = 0 if sex == "Female" else 1
log_pop = np.log(population * 1_000_000)
diag_code = {v: k for k, v in DIAGNOSIS_MAP.items()}[diagnosis]

# ==================== Prediction ====================
with st.spinner('Loading models and calculating predictions...'):
    models, df = load_data()

input_data = pd.DataFrame([[log_pop, sex_code, age_code, year, diag_code]],
                          columns=["log_population", "sex", "age", "year", "diagnosis"])

try:
    predictions = {
        "YLDs": models["YLDs"].predict(input_data)[0],
        "Incidence": models["Incidence"].predict(input_data)[0],
        "Prevalence": models["Prevalence"].predict(input_data)[0]
    }
except Exception as e:
    st.error(f"Prediction failed: {str(e)}")
    st.stop()

# ==================== Results Display ====================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Years Lived with Disability (YLDs)", f"{predictions['YLDs']:,.1f}")
with col2:
    st.metric("Incidence Rate", f"{predictions['Incidence']:.2f}/100,000")
with col3:
    st.metric("Prevalence Rate", f"{predictions['Prevalence']:.2f}%")

# ==================== SHAP Explanation ====================
st.divider()
st.header("🧠 Model Interpretation")

try:
    explainer = shap.Explainer(models["YLDs"])
    shap_values = explainer(input_data)

    plt.figure(figsize=(10, 4))
    shap.plots.bar(shap_values[0], show=False)
    plt.title("Feature Impact Analysis", fontsize=14)
    plt.xlabel("SHAP Value (Impact on YLDs)", fontsize=10)
    st.pyplot(plt.gcf())

    feature_names = ["Log Population", "Gender", "Age Group", "Year", "Diagnosis"]
    df_impact = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0].round(4),
        "Impact Direction": ["Risk Increase" if x > 0 else "Risk Decrease" for x in shap_values.values[0]]
    })

    st.dataframe(
        df_impact,
        column_config={
            "SHAP Value": st.column_config.NumberColumn(format="%.4f"),
            "Impact Direction": st.column_config.SelectboxColumn(options=["Risk Increase", "Risk Decrease"])
        },
        hide_index=True
    )

except Exception as e:
    st.error(f"Model interpretation failed: {str(e)}")

# ==================== Data Validation ====================
with st.expander("🔍 Data Distribution Validation"):
    tab1, tab2 = st.tabs(["Age Distribution", "Gender Distribution"])

    with tab1:
        age_dist = df["age_name"].value_counts().reset_index()
        age_dist.columns = ["age_name", "count"]
        st.bar_chart(age_dist, x="age_name", y="count")

    with tab2:
        sex_dist = df["sex_name"].value_counts().reset_index()
        sex_dist.columns = ["sex_name", "count"]
        st.bar_chart(sex_dist, x="sex_name", y="count")

# ==================== Styling ====================
st.markdown("""
<style>
    [data-testid=stMetric] {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid=stMetricLabel] {
        color: #2E86AB;
        font-weight: bold;
    }
    [data-testid=stMetricValue] {
        color: #333333;
        font-size: 24px !important;
    }
</style>
""", unsafe_allow_html=True)
