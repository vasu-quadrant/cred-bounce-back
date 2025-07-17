from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import boto3
import json
import io
# from scripts.preprocess import preprocess
# from notebooks.preprocessing import apply_preprocessing
from scripts.new_preprocessing import apply_preprocessing
import tempfile
import os
import boto3
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

THRESHOLDS = {
    "Platinum": 0.8,
    "Gold": 0.6,
    "Silver": 0.4,
    "Bronze": 0.2,
    "Copper": 0
}

BUCKET_NAME = "a-sample-bajaj-bucket"
PREFIX = "sample-bajaj-local"
s3_client = boto3.client("s3")

def get_label(score):
    for label, threshold in THRESHOLDS.items():
        if score >= threshold:
            return label
    return "Unknown" 


async def predict_with_file(file: UploadFile, model, preprocessor):#ENDPOINT_NAME):
    """
        Function to process a CSV file and return predictions.
    """

    # Save the uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    file_name = file.filename
   
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Read and preprocess the data
        raw_data = pd.read_csv(temp_file_path)
        print("Raw Data")
        # response = s3_client.get_object(
        #         Bucket = BUCKET_NAME,
        #         Key = F"{PREFIX}/packages/preprocessing_objects.pkl"
        #     )
        # preprocessing_objects_from_s3 = pickle.loads(response["Body"].read())
        print("Preproccessing..")
        # preprocessed_data = preprocess(raw_data)
        # preprocessed_data = apply_preprocessing(raw_data, preprocessing_objects_from_s3)

        # ===========================
        preprocessed_data = apply_preprocessing(raw_data, preprocessor)
        # ===========================

        print(preprocessed_data.shape, preprocessed_data.columns)

        print("Preproccesing done")
        # body = (preprocessed_data.iloc[:, 1:]).to_csv(index=False, header=False)
        # body = body.encode("utf-8")

        
        print("Involking")
        # # Make SageMaker inference call
        # sagemaker_runtime_client = boto3.client("runtime.sagemaker")
        # response = sagemaker_runtime_client.invoke_endpoint(
        #     EndpointName=ENDPOINT_NAME,
        #     ContentType="text/csv",
        #     Body=body,
        #     Accept="application/json"
        # )
 
        # Process the response
        # result = json.loads(response['Body'].read().decode('utf-8'))
 
        # Process scores
        # scores = []
        # ID = []
        # for index, score in enumerate(result["predictions"]):
        #     scores.append(round(score["score"], 3))
        #     ID.append(index)
     
        # ==============
        # score for good 
        for col in preprocessed_data.select_dtypes(include='object').columns:
            preprocessed_data[col] = preprocessed_data[col].astype(float)

        preprocessed_data.columns = preprocessed_data.columns.astype(str).str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
        scores = model.predict_proba(preprocessed_data)[:, 1]
        scores = [round(float(score), 3) for score in scores]
        ID = list(range(1, len(scores) + 1))
        # ==============


        THRESHOLD = 0.5
        dataset = pd.DataFrame({"ID": ID, "Score": scores})
        dataset["Label"] = dataset["Score"].apply(get_label)
     
        # Combine with original data
        result_dataset = pd.concat([raw_data, dataset], axis=1)
       
        # Prepare the response
        summary = dataset["Label"].value_counts().to_dict()
        num_platinum = summary.get('Platinum', 0)
        num_gold = summary.get('Gold', 0)
        num_silver = summary.get('Silver', 0)
        num_bronze = summary.get('Bronze', 0)
        num_copper = summary.get('Copper', 0)


        # Save the rawdata to S3
        csv_buffer = io.StringIO()
        raw_data.to_csv(csv_buffer, index=False, header=True)
        input_Key = f'{PREFIX}/input/rawdata/{file_name}'
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=input_Key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )


        # Save the result to S3
        csv_buffer = io.StringIO()
        result_dataset.to_csv(csv_buffer, index=False, header=True)
        output_Key = f'{PREFIX}/output/result/{file_name}'
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=output_Key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )

       
        return {
            "predictions": result_dataset.to_dict(orient='records'),
            "summary": {
                "platinum_predictions": int(num_platinum),
                "glod_predictions": int(num_gold),
                "silver_predictions": int(num_silver),
                "bronze_predictions": int(num_bronze),
                "copper_predictions": int(num_copper),
                "total_predictions": int(num_platinum + num_gold + num_silver + num_bronze + num_copper)
            }
        }
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
   
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        os.rmdir(temp_dir)