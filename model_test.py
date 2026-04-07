import pickle
import traceback
import sys

print("="*70)
print(" MODEL INSPECTION TOOL ".center(70, "="))
print("="*70)

try:
    # Load model
    print("\n📦 Loading model...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded successfully!\n")
    
    # Basic information
    print("="*70)
    print(" BASIC INFORMATION ".center(70, "-"))
    print("="*70)
    print(f"Type: {type(model)}")
    print(f"Class name: {type(model).__name__}")
    print(f"Module: {type(model).__module__}")
    print(f"String representation: {str(model)[:100]}")
    
    # Check for predict method
    print("\n" + "="*70)
    print(" PREDICTION CAPABILITY ".center(70, "-"))
    print("="*70)
    if hasattr(model, 'predict'):
        print("✓ Has 'predict' method")
        print(f"  Method type: {type(model.predict)}")
        
        # Try to get method signature
        try:
            import inspect
            sig = inspect.signature(model.predict)
            print(f"  Signature: {sig}")
        except:
            pass
    else:
        print("✗ NO 'predict' method found!")
    
    # Check if it's a Pipeline
    print("\n" + "="*70)
    print(" PIPELINE INFORMATION ".center(70, "-"))
    print("="*70)
    
    if hasattr(model, 'steps'):
        print("✓ Has 'steps' attribute (likely a Pipeline)")
        try:
            steps = model.steps
            print(f"  Type of steps: {type(steps)}")
            print(f"  Number of steps: {len(steps)}")
            print("\n  Pipeline Steps:")
            for i, (name, obj) in enumerate(steps, 1):
                print(f"    {i}. {name}")
                print(f"       Type: {type(obj).__name__}")
                print(f"       Module: {type(obj).__module__}")
        except Exception as e:
            print(f"  ✗ Error accessing steps: {e}")
    else:
        print("✗ No 'steps' attribute (not a Pipeline)")
    
    if hasattr(model, 'named_steps'):
        print("\n✓ Has 'named_steps' attribute")
        try:
            named_steps = model.named_steps
            print(f"  Type: {type(named_steps)}")
            
            # Try different ways to access it
            if hasattr(named_steps, 'keys'):
                print(f"  Keys: {list(named_steps.keys())}")
            elif isinstance(named_steps, dict):
                print(f"  Keys (dict): {list(named_steps.keys())}")
            else:
                print(f"  (Cannot get keys - type: {type(named_steps)})")
        except Exception as e:
            print(f"  ✗ Error accessing named_steps: {e}")
    
    # Feature information
    print("\n" + "="*70)
    print(" FEATURE INFORMATION ".center(70, "-"))
    print("="*70)
    
    if hasattr(model, 'feature_names_in_'):
        try:
            features = model.feature_names_in_
            print(f"✓ Expected feature names found ({len(features)} features):")
            for i, feat in enumerate(features[:10], 1):
                print(f"  {i}. {feat}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more features")
        except Exception as e:
            print(f"✗ Error accessing feature_names_in_: {e}")
    else:
        print("✗ No 'feature_names_in_' attribute")
    
    if hasattr(model, 'n_features_in_'):
        try:
            print(f"\n✓ Number of features expected: {model.n_features_in_}")
        except Exception as e:
            print(f"✗ Error accessing n_features_in_: {e}")
    
    # All attributes and methods
    print("\n" + "="*70)
    print(" AVAILABLE ATTRIBUTES & METHODS ".center(70, "-"))
    print("="*70)
    
    all_attrs = dir(model)
    
    # Callable methods
    methods = [attr for attr in all_attrs if not attr.startswith('_') and callable(getattr(model, attr, None))]
    print(f"\n✓ Callable methods ({len(methods)}):")
    for method in methods[:15]:
        print(f"  - {method}")
    if len(methods) > 15:
        print(f"  ... and {len(methods) - 15} more")
    
    # Properties/Attributes
    properties = [attr for attr in all_attrs if not attr.startswith('_') and not callable(getattr(model, attr, None))]
    print(f"\n✓ Properties/Attributes ({len(properties)}):")
    for prop in properties[:15]:
        try:
            value = getattr(model, prop)
            value_str = str(value)[:50]
            print(f"  - {prop}: {value_str}")
        except Exception as e:
            print(f"  - {prop}: <error accessing: {str(e)[:30]}>")
    if len(properties) > 15:
        print(f"  ... and {len(properties) - 15} more")
    
    # Test prediction
    print("\n" + "="*70)
    print(" PREDICTION TEST ".center(70, "-"))
    print("="*70)
    
    import numpy as np
    import pandas as pd
    
    # Test different input formats
    print("\nTesting with different input formats...\n")
    
    # Test 1: Simple NumPy array (8 features)
    print("Test 1: NumPy array (8 basic features)")
    test1 = np.array([[0, 15, 3, 10, 30, 14, 45, 240]])
    print(f"  Shape: {test1.shape}")
    try:
        result = model.predict(test1)
        print(f"  ✓ SUCCESS! Prediction: {result}")
        print(f"  Result type: {type(result)}, Shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)[:100]}")
    
    # Test 2: Pandas DataFrame (8 features with names)
    print("\nTest 2: Pandas DataFrame (8 features)")
    test2 = pd.DataFrame([[0, 15, 3, 10, 30, 14, 45, 240]], 
                         columns=['Total_Stops', 'Journey_day', 'Journey_month', 
                                 'Dep_hour', 'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration'])
    print(f"  Shape: {test2.shape}")
    print(f"  Columns: {list(test2.columns)}")
    try:
        result = model.predict(test2)
        print(f"  ✓ SUCCESS! Prediction: {result}")
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)[:100]}")
    
    # Test 3: With categorical features
    print("\nTest 3: DataFrame with categorical features")
    test3 = pd.DataFrame([{
        'Airline': 'IndiGo',
        'Source': 'Banglore',
        'Destination': 'Delhi',
        'Total_Stops': 0,
        'Journey_day': 15,
        'Journey_month': 3,
        'Dep_hour': 10,
        'Dep_min': 30,
        'Arrival_hour': 14,
        'Arrival_min': 45,
        'Duration': 240
    }])
    print(f"  Shape: {test3.shape}")
    print(f"  Columns: {list(test3.columns)}")
    try:
        result = model.predict(test3)
        print(f"  ✓ SUCCESS! Prediction: {result}")
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)[:100]}")
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY ".center(70, "-"))
    print("="*70)
    print(f"Model Type: {type(model).__name__}")
    print(f"Has predict: {hasattr(model, 'predict')}")
    print(f"Is Pipeline: {hasattr(model, 'steps')}")
    
    if hasattr(model, 'n_features_in_'):
        try:
            print(f"Expected features: {model.n_features_in_}")
        except:
            print(f"Expected features: Unknown")
    
    print("\n💡 Recommendation:")
    if hasattr(model, 'steps'):
        print("  Your model is a Pipeline. Use Strategy 2 or 4 in the Flask app.")
    else:
        print("  Your model is a standalone estimator. Use Strategy 1 or 3.")
    
except FileNotFoundError:
    print("\n✗ ERROR: model.pkl not found!")
    print("  Make sure model.pkl is in the same directory as this script.")
except Exception as e:
    print(f"\n✗ UNEXPECTED ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "="*70)
print(" END OF INSPECTION ".center(70, "="))
print("="*70)