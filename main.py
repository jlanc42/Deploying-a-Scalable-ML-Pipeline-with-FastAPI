import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# TODO: Enter the path for the saved encoder and model
encoder_path = os.path.join("model", "encoder.pkl")  
model_path = os.path.join("model", "model.pkl")  

encoder = load_model(encoder_path)
model = load_model(model_path)

# TODO: Create a RESTful API using FastAPI
app = FastAPI()

# TODO: Create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Welcome message at the root endpoint. """
    return {"message": "Welcome to the Income Prediction API!"}

# TODO: Create a POST on a different path that does model inference
@app.post("/predict/")
async def post_inference(data: Data):
    """
    Perform model inference on the input data.
    
    Args:
        data (Data): Input data from the user in JSON format.
    
    Returns:
        dict: The predicted income label (either <=50K or >50K).
    """
    
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data_cleaned = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data_df = pd.DataFrame.from_dict(data_cleaned)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    # Process the input data using the same encoder and label binarizer used during training.
    data_processed, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        encoder=encoder,
        training=False  # We are doing inference, not training
    )
    
    # Perform inference using the trained model
    prediction = inference(model, data_processed)
    
    # Return the result (either <=50K or >50K)
    return {"result": apply_label(prediction)}