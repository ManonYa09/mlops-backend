from pydantic import BaseModel

class PredictionModel(BaseModel):
    RowNumber:int
    CustomerId:int
    Surnam:str
    CreditScore:int
    Geography:str
    Gender:str
    Age:int
    Tenure:int
    Balance:float
    NumOfProducts:int
    HasCrCard:int
    IsActiveMember:int
    EstimatedSalary:float

    