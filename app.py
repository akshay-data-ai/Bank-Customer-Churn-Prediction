
# Bank Customer Churn Prediction - Streamlit Application

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Step 1: Configure Streamlit page
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Step 2: Load saved Gradient Boosting model
@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name(
        "bank_customer_churn_model.pkl"
    )
    return joblib.load(model_path)

bundle = load_model()
model = bundle['model']
threshold = bundle['threshold']

# Step 3: Professional Dashboard Header

st.title("🏦 Banking Intelligence Dashboard")

st.markdown(
    "### AI-Powered Customer Churn Prediction"
)

st.write(
    "Analyze customer profiles and estimate churn risk "
    "using our trained Gradient Boosting model."
)

st.divider()

# Model Performance KPI Cards
st.subheader("📊 Model Performance Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model Accuracy",
        value="85.30%"
    )

with col2:
    st.metric(
        label="Model Recall",
        value="65.36%"
    )

with col3:
    st.metric(
        label="Model F1-Score",
        value="64.41%"
    )

st.caption(
    "Metrics are from exploratory test evaluation."
)

st.info(
    "🤖 Selected Model: Gradient Boosting | "
    f"Prediction Threshold: {threshold * 100:.0f}%"
)

st.divider()

# Step 4: Customer details input form
st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score", 300, 850, 650
        )

        country = st.selectbox(
            "Country",
            bundle['country_categories'],
            index=1
        )

        gender = st.selectbox(
            "Gender",
            bundle['gender_categories']
        )

        age = st.number_input(
            "Age", 18, 100, 45
        )

        tenure = st.number_input(
            "Tenure (Years)", 0, 10, 5
        )

    with col2:
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=100000.0
        )

        products_number = st.number_input(
            "Number of Products", 1, 4, 1
        )

        credit_card = st.selectbox(
            "Has Credit Card?", ["Yes", "No"]
        )

        active_member = st.selectbox(
            "Active Member?", ["No", "Yes"]
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

# Step 5: Prepare customer data and predict
if submitted:

    customer = pd.DataFrame([{
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': 1 if credit_card == "Yes" else 0,
        'active_member': 1 if active_member == "Yes" else 0,
        'estimated_salary': estimated_salary
    }])

    # Apply the same encoding used during training
    customer['country'] = pd.Categorical(
        customer['country'],
        categories=bundle['country_categories']
    )

    customer['gender'] = pd.Categorical(
        customer['gender'],
        categories=bundle['gender_categories']
    )

    customer_encoded = pd.get_dummies(
        customer,
        columns=['country', 'gender'],
        drop_first=True,
        dtype=int
    )

    # Match model feature columns and order
    customer_encoded = customer_encoded.reindex(
        columns=bundle['feature_columns'],
        fill_value=0
    )

    # Predict churn probability
    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    prediction = int(probability >= threshold)

    
    # Step 6: Professional Prediction Results
    st.divider()

    st.subheader("🔍 Customer Churn Risk Assessment")

    with st.container(border=True):

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Estimated Churn Score",
                value=f"{probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                label="Decision Threshold",
                value=f"{threshold * 100:.0f}%"
            )

        st.progress(
            float(probability),
            text="Customer Churn Risk Score"
        )

        st.write("### Prediction Outcome")

        if prediction == 1:
            st.warning(
                "⚠️ Customer Likely to Leave the Bank"
            )

            st.write(
                "The model estimates a churn score "
                "above the selected threshold. "
                "This customer may warrant further "
                "retention analysis."
            )

        else:
            st.success(
                "✅ Customer Likely to Stay with the Bank"
            )

            st.write(
                "The model estimates a churn score "
                "below the selected threshold."
            )

    st.caption(
        "Educational model estimate only. "
        "Probabilities are not independently calibrated "
        "or validated for real-world banking decisions."
    )
# Step 7: Footer
st.divider()
st.caption(
    "Bank Customer Churn Prediction | Machine Learning Project"
)
