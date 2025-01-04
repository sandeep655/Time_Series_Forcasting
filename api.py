# Flask API Implementation
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the saved Prophet model at the start of the app
# with open('FurnitureSalesForecast/notebook/prophet_model.pkl', 'rb') as f:
#     prophet_model = pickle.load(f)
with open('prophet_model.pkl', 'rb') as f:
    prophet_model = pickle.load(f)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        periods = data.get('periods', 12)

        # Log input data
        print(f"Request data: {data}")

        # Generate future dates and forecast
        future = prophet_model.make_future_dataframe(periods=periods, freq='M')
        forecast = prophet_model.predict(future)

        # Format date and round sales to 2 decimal places
        forecast['ds'] = forecast['ds'].dt.date
        forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].round(2)

        # Log the forecast data
        print(f"Generated forecast: {forecast[['ds', 'yhat']].tail(periods)}")

        # Return the forecast as JSON
        return jsonify(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient='records'))
    except Exception as e:
        print(f"Error in forecast endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)