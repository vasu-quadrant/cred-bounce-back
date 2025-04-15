import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2


def split_personal_status(status):
    parts = status.split(' ', 1)
    gender = parts[0]
    marital_status = parts[1] if len(parts) > 1 else ''
    return pd.Series([gender, marital_status])


def engineer_features(df):
    df = df.copy()
    if 'Customer_ID' in df.columns:
        df = df.drop(columns=['Customer_ID'])

    df[['gender', 'marital_status']] = df['personal_status'].apply(split_personal_status)
    df.drop(columns=['personal_status'], inplace=True)

    job_map = {
        'high qualif/self emp/mgmt': 4,
        'skilled': 3,
        'unskilled resident': 2,
        'unemp/unskilled non res': 1
    }

    if 'job' in df.columns:
        df['job'] = df['job'].map(job_map)

    if 'credit_amount' in df.columns and 'job' in df.columns:
        df['credit_job_ratio'] = df['credit_amount'] / df['job'].replace(0, 1)

    if 'age' in df.columns:
        df['credit_age_ratio'] = df['credit_amount'] / df['age']

    if 'duration' in df.columns:
        df['monthly_burden'] = df['credit_amount'] / df['duration']

    if 'installment_commitment' in df.columns and 'existing_credits' in df.columns:
        df['debt_burden'] = df['installment_commitment'] * df['existing_credits']

    return df


def build_preprocessor(df, encoding_method='onehot', drop_first=False, handle_unknown='ignore'):
    """Create preprocessing transformer and extract transformed feature names."""
    binary_mappings = {
        'gender': {'male': 1, 'female': 0},
        'own_telephone': {'yes': 1, 'none': 0},
        'foreign_worker': {'yes': 1, 'no': 0},
        'class': {'good': 1, 'bad': 0}
    }

    binary_cols = ['own_telephone', 'foreign_worker', 'class', 'gender']
    multi_category_cols = [
        'checking_status', 'credit_history', 'purpose', 'savings_status',
        'employment', 'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'marital_status'
    ]
    numerical_cols = [
        'duration', 'credit_amount', 'installment_commitment',
        'residence_since', 'age', 'existing_credits','job', 'num_dependents',
        'credit_job_ratio', 'credit_age_ratio', 'monthly_burden', 'debt_burden'
    ]

    numerical_cols = [col for col in numerical_cols if col in df.columns]
    multi_category_cols = [col for col in multi_category_cols if col in df.columns]
    binary_cols = [col for col in binary_cols if col in df.columns]

    for col in binary_cols:
        df[col] = df[col].map(binary_mappings.get(col, {}))

    for col in ['credit_amount', 'age', 'credit_job_ratio', 'credit_age_ratio', 'monthly_burden']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    if encoding_method == 'onehot':
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first' if drop_first else None,
                                     handle_unknown=handle_unknown,
                                     sparse_output=False))
        ])
    else:
        raise ValueError("Only onehot encoding is supported.")

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, multi_category_cols)
    ], remainder='passthrough')

    return preprocessor, numerical_cols, multi_category_cols, binary_cols


def fit_full_pipeline(df, top_k=60, p_thresh=0.05):
    """Fit the full pipeline and return transformed data + reusable preprocessor."""
    df = engineer_features(df)
    preprocessor, num_cols, cat_cols, bin_cols = build_preprocessor(df)

    X_transformed = preprocessor.fit_transform(df)
    feature_names = num_cols.copy()

    onehot = preprocessor.named_transformers_['cat']['onehot']
    if hasattr(onehot, 'get_feature_names_out'):
        feature_names += list(onehot.get_feature_names_out(cat_cols))
    else:
        feature_names += list(onehot.get_feature_names(cat_cols))

    feature_names += bin_cols

    transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)

    # Chi-square feature selection
    X = transformed_df.drop(columns=['class'])
    y = transformed_df['class']

    neg_cols = X.loc[:, (X < 0).any()].columns
    pos_cols = X.loc[:, (X >= 0).all()].columns

    k_best = min(top_k, len(pos_cols))
    chi2_selector = SelectKBest(chi2, k=k_best)
    chi2_selector.fit(X[pos_cols], y)

    chi2_df = pd.DataFrame({
        "Feature": pos_cols,
        "Chi2 Score": chi2_selector.scores_,
        "P-Value": chi2_selector.pvalues_
    })

    sig_features = chi2_df[chi2_df["P-Value"] < p_thresh]["Feature"].tolist()
    selected_features = list(neg_cols) + sig_features

    final_df = pd.concat([X[selected_features], y], axis=1)

    # Store feature list in preprocessor
    preprocessor.chi_square_selected_features = selected_features
    preprocessor.all_transformed_features = feature_names

    return final_df, preprocessor


def apply_preprocessing(new_df, preprocessor):
    """Apply previously-fitted preprocessing on new data."""
    new_df = engineer_features(new_df)
    # print(new_df.columns)
    # print(new_df.to_json(orient= 'records'))
    # print(new_df.shape, new_df.columns)

    binary_mappings = {
        'gender': {'male': 1, 'female': 0},
        'own_telephone': {'yes': 1, 'none': 0},
        'foreign_worker': {'yes': 1, 'no': 0},
        'class': {'good': 1, 'bad': 0}
    }

    binary_cols = ['own_telephone', 'foreign_worker', 'class', 'gender']
    binary_cols = [col for col in binary_cols if col in new_df.columns]

    for col in binary_cols:
        new_df[col] = new_df[col].map(binary_mappings.get(col, {}))

    for col in ['credit_amount', 'age', 'credit_job_ratio', 'credit_age_ratio', 'monthly_burden']:
        if col in new_df:
            new_df[col] = np.log1p(new_df[col])

    X_transformed = preprocessor.transform(new_df)
    print(X_transformed)


    transformed_df = pd.DataFrame(X_transformed, columns=preprocessor.all_transformed_features, index=new_df.index)

    print(transformed_df)


    selected_cols = preprocessor.chi_square_selected_features
    return transformed_df[selected_cols]

# Example usage:
if __name__ == "__main__":
    raw_data = pd.read_csv("raw_data1 (1).csv")
    final_df, preprocessor = fit_full_pipeline(raw_data)