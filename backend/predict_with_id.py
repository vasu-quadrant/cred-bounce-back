import pandas as pd
import boto3
import json
# from notebooks.preprocessing import apply_preprocessing
# from scripts.preprocess import preprocess
from scripts.new_preprocessing import apply_preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

s3_client = boto3.client("s3")

BUCKET_NAME = "a-sample-bajaj-bucket"
PREFIX = "sample-bajaj-local"


def predict_with_ID(customerID , model, preprocessor):# ENDPOINT_NAME):
    """
        Function to process a CSV file with Customer ID and return Category.
    """
    print(f'{PREFIX}/raw/raw_data1.csv')
    s3_response = s3_client.get_object(Bucket=BUCKET_NAME, Key=f'{PREFIX}/raw/raw_data1.csv')
    print("Fetched from s3")

    # Read CSV content into DataFrame
    raw_data = pd.read_csv(s3_response['Body'])    
    customer_values = raw_data[raw_data["Customer_ID"] == customerID]
    print("Call preproessing - ", customer_values.shape)


    # response = s3_client.get_object(
    #             Bucket = BUCKET_NAME,
    #             Key = F"{PREFIX}/packages/preprocessing_objects.pkl"
    #         )
    # preprocessing_objects_from_s3 = pickle.loads(response["Body"].read())

    # preprocessed_data = preprocess(customer_values)
    # print(type(preprocessing_objects_from_s3))
    # print(preprocessing_objects_from_s3)
    # print("Predictions: ", preprocessing_objects_from_s3["preprocessor"])
    # preprocessed_data = apply_preprocessing(customer_values, preprocessing_objects_from_s3)

    # ================================
    preprocessed_data = apply_preprocessing(customer_values, preprocessor)
    # ================================
    print(preprocessed_data.shape, preprocessed_data.columns)

    # body = (preprocessed_data.iloc[:, 1:]).to_csv(index=False, header=False)
    # body = body.encode("utf-8")

    print("Invoking")
    # # Make SageMaker inference call
    # sagemaker_runtime_client = boto3.client("runtime.sagemaker")
    # response = sagemaker_runtime_client.invoke_endpoint(
    #     EndpointName=  ENDPOINT_NAME,
    #     ContentType="text/csv",
    #     Body=body,
    #     Accept="application/json"
    # )

    # # Process the response
    # result = json.loads(response['Body'].read().decode('utf-8'))

    # score = result["predictions"][0]['score']
    # print(result, score)

    # =================================
    for col in preprocessed_data.select_dtypes(include='object').columns:
        preprocessed_data[col] = preprocessed_data[col].astype(float)

    preprocessed_data.columns = preprocessed_data.columns.astype(str).str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    score = model.predict_proba(preprocessed_data)[0][1]
    # ================================

    print("Returning Results from predict with ID fucntion")

    return {
        "customer_prediction": {
            "customer_ID": customerID,
            "data": customer_values.to_dict(orient='records'),
            "score": round(float(score), 3)
        }
    }
            
if __name__ == "__main__":
    predict_with_ID("CUST-49028", "end-point-XGBoost-cred-clf-2025-03-14-10-43-30")



