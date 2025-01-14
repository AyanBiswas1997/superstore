
from src.superstore_ml.pipelines.prediction_pipeline import CustomData, PredictPipeline
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template("form.html")
    else:
        
            # Collecting data with default values for missing or invalid inputs
            data = CustomData(
                Item_Weight=float(request.form.get('Item_Weight')),
                Item_Fat_Content=request.form.get('Item_Fat_Content'),
                Item_Visibility=float(request.form.get('Item_Visibility')),
                Item_MRP=float(request.form.get('Item_MRP')),
                Outlet_Establishment_Year=float(request.form.get('Outlet_Establishment_Year')),
                Item_Type=request.form.get('Item_Type'),
                Outlet_Size=request.form.get('Outlet_Size'),
                Outlet_Location_Type=request.form.get('Outlet_Location_Type'),
                Outlet_Type=request.form.get('Outlet_Type')
            )

            # Prepare final data for prediction
            final_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            result = round(pred[0], 2)

            return render_template('result.html', prediction=result)

        

if __name__ == '__main__':
    app.run()



        
        
    

          
            

        
