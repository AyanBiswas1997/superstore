
from src.superstore_ml.pipelines.prediction_pipeline import CustomData
predict_data=CustomData(9.300, 'Low Fat', 0.016047, 249.8092, 1999, 'Dairy', 'Medium', 'Tier 1', 'Supermarket Type1')
data=predict_data.get_data_as_dataframe()
print(data)