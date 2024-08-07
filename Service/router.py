import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from Service.load_model import load_best_model_and_scaler

router = APIRouter(prefix="/group_18",
                   tags=["Group 18 assignment 1"])


class ModelInput(BaseModel):
    model_input: dict


@router.post("/predict")
def invoke_model(InferenceInput: ModelInput):
    model, scaler = load_best_model_and_scaler(is_local_model=True)

    input_json = InferenceInput.model_input
    df = pd.DataFrame.from_dict(input_json, orient="index").T

    # Use the pre-loaded scaler
    preprocessed_input = scaler.transform(df)
    X_test_final = pd.DataFrame(preprocessed_input, columns=df.columns)

    # Use the pre-loaded model
    prediction = model.predict(X_test_final)

    if prediction[0] == 1:
        liver_disease = "Yes"
    else:
        liver_disease = "No"

    return {"Liver Disease": liver_disease}
