from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import traceback
import sys

app = Flask(__name__)

# Global variables
model = None
model_type = None

# At the top of app.py, after loading model
try:
    with open('metrics.pkl', 'rb') as f:
        MODEL_METRICS = pickle.load(f)
    print("✓ Metrics loaded successfully!")
except:
    print("⚠️ Metrics file not found, using defaults")
    MODEL_METRICS = {
        'model_name': 'XGBRegressor',
        'training_score': 0.9282644742578454,
        'r2_score': 0.8337433041016059,
        'mse': 3236618.7974470966,
        'mae': 1146.576267986519,
        'rmse': 1799.060531901886,
        'mape': 13.022283990094472,
        'sample_predictions': [16627.258, 6014.878, 7973.472, 3526.002, 7921.331, 6444.837]
    }

def load_model():
    """Load the model with proper instantiation"""
    global model, model_type
    try:
        with open('model.pkl', 'rb') as file:
            loaded_obj = pickle.load(file)
        
        # Check if it's a class or instance
        if isinstance(loaded_obj, type):
            print("⚠️ Loaded object is a class, attempting to instantiate...")
            model = loaded_obj()  # Instantiate the class
        else:
            model = loaded_obj
        
        model_type = type(model).__name__
        print("="*60)
        print(f"✓ Model loaded successfully!")
        print(f"✓ Model Type: {model_type}")
        print(f"✓ Model Class: {model.__class__}")
        
        # Check attributes
        if hasattr(model, 'n_features_in_'):
            print(f"✓ Number of features: {model.n_features_in_}")
        
        if hasattr(model, 'feature_names_in_'):
            print(f"✓ Feature names: {list(model.feature_names_in_)[:5]}...")
        
        if hasattr(model, 'steps'):
            print(f"✓ Pipeline steps:")
            for name, step in model.steps:
                print(f"  - {name}: {type(step).__name__}")
        
        print("="*60)
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        traceback.print_exc()
        return False

# Load model
load_model()


# Model Performance Metrics (from training)
MODEL_METRICS = {
    'model_name': 'XGBRegressor',
    'training_score': 0.9282644742578454,
    'r2_score': 0.8337433041016059,
    'mse': 3236618.7974470966,
    'mae': 1146.576267986519,
    'rmse': 1799.060531901886,
    'mape': 13.022283990094472,
    'sample_predictions': [16627.258, 6014.878, 7973.472, 3526.002, 7921.331, 6444.837]
}



# Categorical values
AIRLINES = [
    'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 
    'Jet Airways Business', 'Multiple carriers', 
    'Multiple carriers Premium economy', 'SpiceJet', 
    'Trujet', 'Vistara', 'Vistara Premium economy'
]

SOURCES = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
DESTINATIONS = ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']
STOPS = ['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']


def preprocess_input(data):
    """
    Comprehensive preprocessing that tries multiple formats
    """
    # Extract and convert data
    airline = data.get('airline')
    source = data.get('source')
    destination = data.get('destination')
    total_stops = data.get('total_stops')
    
    journey_day = int(data.get('journey_day'))
    journey_month = int(data.get('journey_month'))
    dep_hour = int(data.get('dep_hour'))
    dep_min = int(data.get('dep_min', 0))
    arrival_hour = int(data.get('arrival_hour'))
    arrival_min = int(data.get('arrival_min', 0))
    duration_hours = int(data.get('duration_hours', 0))
    duration_mins = int(data.get('duration_mins', 0))
    
    # Calculate duration
    duration_total_mins = duration_hours * 60 + duration_mins
    
    # Convert stops
    stops_dict = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    stops_num = stops_dict.get(total_stops, 0)
    
    # Try to match expected features
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
        print(f"Model expects these features: {feature_names}")
        
        # Build feature dictionary
        feature_dict = {}
        
        for feat in feature_names:
            # Numerical features
            if 'Total_Stops' in feat or feat == 'Total_Stops':
                feature_dict[feat] = stops_num
            elif 'Journey_day' in feat or feat == 'Journey_day':
                feature_dict[feat] = journey_day
            elif 'Journey_month' in feat or feat == 'Journey_month':
                feature_dict[feat] = journey_month
            elif 'Dep_hour' in feat or feat == 'Dep_hour':
                feature_dict[feat] = dep_hour
            elif 'Dep_min' in feat or feat == 'Dep_min':
                feature_dict[feat] = dep_min
            elif 'Arrival_hour' in feat or feat == 'Arrival_hour':
                feature_dict[feat] = arrival_hour
            elif 'Arrival_min' in feat or feat == 'Arrival_min':
                feature_dict[feat] = arrival_min
            elif 'Duration_hours' in feat or feat == 'Duration_hours':
                feature_dict[feat] = duration_hours
            elif 'Duration_mins' in feat or feat == 'Duration_mins':
                feature_dict[feat] = duration_mins
            elif 'Duration_total_mins' in feat or feat == 'Duration_total_mins' or feat == 'Duration':
                feature_dict[feat] = duration_total_mins
            
            # Airline one-hot encoding
            elif 'Airline_' in feat:
                airline_name = feat.replace('Airline_', '')
                feature_dict[feat] = 1 if airline == airline_name else 0
            
            # Source one-hot encoding
            elif 'Source_' in feat:
                source_name = feat.replace('Source_', '')
                feature_dict[feat] = 1 if source == source_name else 0
            
            # Destination one-hot encoding
            elif 'Destination_' in feat:
                dest_name = feat.replace('Destination_', '')
                feature_dict[feat] = 1 if destination == dest_name else 0
            
            else:
                # Default to 0
                feature_dict[feat] = 0
        
        # Create DataFrame with exact feature order
        df = pd.DataFrame([feature_dict])
        return df
    
    else:
        # Fallback: Create features with common structure
        features = {
            'Total_Stops': stops_num,
            'Journey_day': journey_day,
            'Journey_month': journey_month,
            'Dep_hour': dep_hour,
            'Dep_min': dep_min,
            'Arrival_hour': arrival_hour,
            'Arrival_min': arrival_min,
            'Duration_hours': duration_hours,
            'Duration_mins': duration_mins,
            'Duration_total_mins': duration_total_mins
        }
        
        # Add one-hot encoded features
        for air in AIRLINES:
            features[f'Airline_{air}'] = 1 if air == airline else 0
        
        for src in SOURCES:
            features[f'Source_{src}'] = 1 if src == source else 0
        
        for dest in DESTINATIONS:
            features[f'Destination_{dest}'] = 1 if dest == destination else 0
        
        df = pd.DataFrame([features])
        
        # Sort columns alphabetically
        df = df.reindex(sorted(df.columns), axis=1)
        
        return df


@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html', 
                         airlines=AIRLINES,
                         sources=SOURCES,
                         destinations=DESTINATIONS,
                         stops=STOPS,
                         metrics=MODEL_METRICS)  # Add this


@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions"""
    try:
        if model is None:
            return render_template('index.html',
                             airlines=AIRLINES,
                             sources=SOURCES,
                             destinations=DESTINATIONS,
                             stops=STOPS,
                             prediction=formatted_price,
                             form_data=form_data,
                             metrics=MODEL_METRICS)  # Add this
        
        # Get form data
        form_data = request.form.to_dict()
        print("\n" + "="*60)
        print("Prediction Request")
        print("="*60)
        print(f"Input data: {form_data}")
        
        # Validate
        required = ['airline', 'source', 'destination', 'total_stops',
                   'journey_day', 'journey_month', 'dep_hour', 
                   'arrival_hour', 'duration_hours']
        
        missing = [f for f in required if not form_data.get(f)]
        if missing:
            return render_template('index.html',
                                 airlines=AIRLINES,
                                 sources=SOURCES,
                                 destinations=DESTINATIONS,
                                 stops=STOPS,
                                 error=f'Missing: {", ".join(missing)}',
                                 form_data=form_data)
        
        # Preprocess
        processed_data = preprocess_input(form_data)
        print(f"\nProcessed data shape: {processed_data.shape}")
        print(f"Features: {list(processed_data.columns)[:5]}...")
        print(f"Sample values: {processed_data.iloc[0][:5].to_dict()}")
        
        # Predict
        prediction = model.predict(processed_data)
        predicted_price = round(float(prediction[0]), 2)
        
        formatted_price = f"₹ {predicted_price:,.2f}"
        
        print(f"\n✓ Prediction successful: {formatted_price}")
        print("="*60 + "\n")
        
        return render_template('index.html',
                             airlines=AIRLINES,
                             sources=SOURCES,
                             destinations=DESTINATIONS,
                             stops=STOPS,
                             prediction=formatted_price,
                             form_data=form_data)
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Error: {error_msg}")
        traceback.print_exc()
        print()
        
        return render_template('index.html',
                             airlines=AIRLINES,
                             sources=SOURCES,
                             destinations=DESTINATIONS,
                             stops=STOPS,
                             error=f'Error: {error_msg}',
                             form_data=request.form.to_dict(),
                             metrics=MODEL_METRICS)  # Add this


@app.route('/debug')
def debug():
    """Debug endpoint"""
    if model is None:
        return {"error": "Model not loaded"}
    
    info = {
        "model_type": model_type,
        "model_class": str(model.__class__),
    }
    
    if hasattr(model, 'n_features_in_'):
        try:
            info["n_features"] = int(model.n_features_in_)
        except:
            info["n_features"] = "unavailable"
    
    if hasattr(model, 'feature_names_in_'):
        try:
            info["feature_names"] = list(model.feature_names_in_)
        except:
            info["feature_names"] = "unavailable"
    
    if hasattr(model, 'steps'):
        try:
            info["pipeline_steps"] = [
                {"name": name, "type": type(obj).__name__}
                for name, obj in model.steps
            ]
        except:
            pass
    
    return info

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Flight Price Prediction App")
    print("="*60)
    
    if model:
        print("✓ Ready!")
        print(f"✓ Model: {model_type}")
    else:
        print("✗ Model not loaded")
    
    print("="*60 + "\n")
    
    # For production
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)