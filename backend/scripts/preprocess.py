import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import boto3


def split_personal_status(status):
    """Splitting the personal_status attribute = gender + marital status"""
    parts = status.split(' ', 1)  # Split on first space only
    gender = parts[0]  # 'male' or 'female'
    marital_status = parts[1] if len(parts) > 1 else ''  # Remainder
    return pd.Series([gender, marital_status])

def preprocess(df):

    df = df.drop(columns = ["Customer_ID"])

    df[['gender', 'marital_status']] = df['personal_status'].apply(split_personal_status)
    df = df.drop(columns= ['personal_status'])

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    true_binary_cols = ['gender', 'own_telephone', 'foreign_worker', 'class']

    # Binary mappings (adjusted 'class' for risk prediction: bad=1, good=0)
    binary_mappings = {
        'gender': {'male': 1, 'female': 0},
        'own_telephone': {'yes': 1, 'none': 0},
        'foreign_worker': {'yes': 1, 'no': 0},
        'class': {'good': 0, 'bad': 1}  # Flipped for consistency with risk prediction
    }
    for col in true_binary_cols:
        df[col] = df[col].map(binary_mappings[col])

    multi_category_cols = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 
                       'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 
                       'marital_status']

    encoded_df = pd.get_dummies(df, columns=multi_category_cols)

    bool_cols = encoded_df.select_dtypes(include=['bool']).columns
    encoded_df[bool_cols] = encoded_df[bool_cols].astype('int8')
    encoded_df['credit_amount'] = np.log1p(encoded_df['credit_amount'])

    scaler = StandardScaler()
    encoded_df[numerical_cols] = scaler.fit_transform(encoded_df[numerical_cols])

    import pickle

    pickle_data = pickle.dumps(scaler)
    s3_client = boto3.client("s3")

    print("Ready to Store")
    s3_client.put_object(
                Body = pickle_data,
                Bucket = "a-sample-bajaj-bucket",
                Key = "sample-bajaj-local/packages/scaler.pkl" 
            )

    
    return encoded_df



def apply_preprocess(df):

    df[['gender', 'marital_status']] = df['personal_status'].apply(split_personal_status)
    df = df.drop(columns= ["Customer_ID", "personal_status", "class"])

    binary_mappings = {
        'gender': {'male': 1, 'female': 0},
        'own_telephone': {'yes': 1, 'none': 0},
        'foreign_worker': {'yes': 1, 'no': 0},
        'class': {'good': 0, 'bad': 1}  # Flipped for consistency with risk prediction
    }

    true_binary_cols = ['gender', 'own_telephone', 'foreign_worker', 'class']
    for col in true_binary_cols:
        df[col] = df[col].map(binary_mappings[col])

    exist_columns = [   
                        'checking_status', 'duration', 'credit_history', 'gender', 'marital_status',
                        'purpose', 'credit_amount', 'savings_status', 'employment',
                        'installment_commitment', 'personal_status', 'other_parties',
                        'residence_since', 'property_magnitude', 'age', 'other_payment_plans',
                        'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone',
                        'foreign_worker'
                    ]

    total_columns = [
                        'duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 
                        'num_dependents', 'own_telephone', 'foreign_worker', 'gender',
                        'checking_status_0<=X<200', 'checking_status_<0', 'checking_status_>=200', 'checking_status_no checking',
                        'credit_history_all paid', 'credit_history_critical/other existing credit', 'credit_history_delayed previously', 'credit_history_existing paid', 'credit_history_no credits/all paid',
                        'purpose_business', 'purpose_domestic appliance', 'purpose_education', 'purpose_furniture/equipment', 'purpose_new car',
                        'purpose_other', 'purpose_radio/tv', 'purpose_repairs', 'purpose_retraining', 'purpose_used car',
                        'savings_status_100<=X<500', 'savings_status_500<=X<1000', 'savings_status_<100', 'savings_status_>=1000', 'savings_status_no known savings',
                        'employment_1<=X<4', 'employment_4<=X<7', 'employment_<1', 'employment_>=7', 'employment_unemployed',
                        'other_parties_co applicant', 'other_parties_guarantor', 'other_parties_none',
                        'property_magnitude_car', 'property_magnitude_life insurance', 'property_magnitude_no known property', 'property_magnitude_real estate',
                        'other_payment_plans_bank', 'other_payment_plans_none', 'other_payment_plans_stores',
                        'housing_for free', 'housing_own', 'housing_rent',
                        'job_high qualif/self emp/mgmt', 'job_skilled', 'job_unemp/unskilled non res', 'job_unskilled resident',
                        'marital_status_div/dep/mar', 'marital_status_div/sep', 'marital_status_mar/wid', 'marital_status_single'
                    ]
    
