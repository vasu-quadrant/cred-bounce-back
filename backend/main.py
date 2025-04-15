
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from predict_with_file import predict_with_file
from predict_with_id import predict_with_ID
import joblib
 
app = FastAPI(title="Cred Prediction API")
 
# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Constants
ENDPOINT_NAME =  "end-point-xg-boost-cred-clf-2025-03-17-16-41-46" 


preprocessor = joblib.load('preprocessor.pkl')
print("Preprocessor Object is Loaded")

model = joblib.load('model2.pkl')

print("Server started")



 
@app.post("/predict")
async def predict(file: UploadFile = File(None), customerID: str = None):
    """
    Endpoint to process CSV files and return predictions.
    Either a CSV file or a customerID must be provided.
    """
    # Check if neither parameter is provided
    print("Called")
    if file is None and customerID is None:
        print("No File and CustomerID")
        raise HTTPException(
            status_code=400,
            detail="Either a file or customerID must be provided."
        )
   
    # If customerID is provided, use ID-based prediction
    if customerID is not None:
        print("Into the CustomerID")
        response = predict_with_ID(customerID, model, preprocessor) #ENDPOINT_NAME)
        print(response)
        print("\n\n===============\n", type(response))
        return response
   
    # At this point, we know file is not None
    if not file.filename.endswith('.csv'):
        print("Into CSV File")
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )
   
    reponse = await predict_with_file(file, model, preprocessor)#ENDPOINT_NAME)
    return reponse
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)