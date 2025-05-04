from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle
import os

class InputPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to prepare input data for model prediction.
    """
    def __init__(self):
        
        print("Pre-processing the input:")
        print('*'*50)

        self.material_columns = [
            'Powder_Material_Group_Cobalt_based',
            'Powder_Material_Group_Copper_based',
            'Powder_Material_Group_Nickel_based',
            'Powder_Material_Group_Steel_based',
            'Powder_Material_Group_Titanium_based',
            'Substrate_Material_Group_Nickel_based',
            'Substrate_Material_Group_Steel_based',
            'Substrate_Material_Group_Titanium_based'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
            
        # Initialize material columns with zeros
        for col in self.material_columns:
            X[col] = 0
            
        # Set appropriate material columns based on input materials
        if 'powder_material' in X.columns:
            for idx, row in X.iterrows():
                powder_col = f"Powder_Material_Group_{row['powder_material']}"
                if powder_col in self.material_columns:
                    X.at[idx, powder_col] = 1
            print("Powder Material encoded")
                    
        if 'substrate_material' in X.columns:
            for idx, row in X.iterrows():
                substrate_col = f"Substrate_Material_Group_{row['substrate_material']}"
                if substrate_col in self.material_columns:
                    X.at[idx, substrate_col] = 1
            print("Substrate Material encoded")

        # Select and order features before ratio calculation
        features_before_ratio = [
            'Power (W)', 'Mass Flowrate (g/min)', 'Travel Velocity (mm/min)',
            'Height (mm)', 'Contact Angle (deg)', 'Spot Size (mm)', 'Width (mm)'
        ]
        
        transform_ready_features = features_before_ratio + self.material_columns
        
        return X[transform_ready_features]
    

class SavedBoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that uses a pre-saved Box-Cox transformer and handles ratio calculation.
    """
    def __init__(self, transformer_path='./transformer/box_cox.pkl'):
        with open(transformer_path, 'rb') as f:
            self.transformer = pickle.load(f)
        print("Box-Cox Transformer object loaded")
            
        # Define columns that need transformation
        self.columns_to_transform = [
            'Power (W)', 
            'Mass Flowrate (g/min)', 
            'Travel Velocity (mm/min)',
            'Height (mm)', 
            'Spot Size (mm)',
            'Width (mm)'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Transform features using saved transformer
        features_array = X_transformed[self.columns_to_transform].values
        transformed_features = self.transformer.transform(features_array)
        
        # Update transformed values
        for idx, col in enumerate(self.columns_to_transform):
            X_transformed[col] = transformed_features[:, idx]
        
        # Calculate Size_Width_Ratio after transformation
        X_transformed['Size_Width_Ratio'] = X_transformed['Spot Size (mm)'] / X_transformed['Width (mm)']
        print("Size-width ratio calculated after transformation")
        
        # Remove original size and width columns as they're not used in final prediction
        X_transformed = X_transformed.drop(['Spot Size (mm)', 'Width (mm)'], axis=1)
        
        # Ensure correct final column order
        final_features = [
            'Power (W)', 'Mass Flowrate (g/min)', 'Travel Velocity (mm/min)',
            'Height (mm)', 'Contact Angle (deg)', 'Size_Width_Ratio'
        ] + [col for col in X_transformed.columns if 'Material_Group' in col]

        # Create a dictionary to store the formatted values
        formatted_values = {}
        for col in final_features:
            value = X_transformed[col].values[0]  # Get the first (and only) value
            if isinstance(value, (int, float)):
                print(f"{col:<40}: {value:>10.4f}")
                formatted_values[col] = round(value, 4)
            else:
                print(f"{col:<40}: {value:>10}")
                formatted_values[col] = value
        print("-" * 50)

        # Create DataFrame from the formatted values
        transformed_df = pd.DataFrame([formatted_values])
        
        return X_transformed[final_features], transformed_df
    
class AspectRatioPredictionPipeline():
    """
    Complete pipeline for aspect ratio prediction using pre-saved transformers
    and models. This pipeline ensures consistent data processing by using
    saved transformers and handling derived features appropriately.
    """
    def __init__(self, model_dir='saved_models', boxcox_path='./transformer/box_cox.pkl'):
        self.model_dir = model_dir
        self.boxcox_path = boxcox_path
        
        # First load the available models
        self.available_models = self._load_available_models()
        
        # Define the expected feature order for our models
        # This is crucial for ensuring predictions are made with features in the correct order
        self.numeric_features = [
            'Power (W)', 
            'Mass Flowrate (g/min)', 
            'Travel Velocity (mm/min)',
            'Height (mm)', 
            'Contact Angle (deg)',
            'Size_Width_Ratio'
        ]
        
        self.material_features = [
            'Powder_Material_Group_Cobalt_based',
            'Powder_Material_Group_Copper_based',
            'Powder_Material_Group_Nickel_based',
            'Powder_Material_Group_Steel_based',
            'Powder_Material_Group_Titanium_based',
            'Substrate_Material_Group_Nickel_based',
            'Substrate_Material_Group_Steel_based',
            'Substrate_Material_Group_Titanium_based'
        ]
        
        # Create the pipeline with proper feature ordering
        self.preprocessing_pipeline = Pipeline([
            ('input_processor', InputPreprocessor()),
            ('box_cox', SavedBoxCoxTransformer(transformer_path=boxcox_path))
        ])
    
    def _load_available_models(self):
        """
        Load all available models from the saved_models directory.
        Includes error handling for missing or corrupted model files.
        """
        models = {}
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory {self.model_dir} not found")
            
        for filename in os.listdir(self.model_dir):
            if filename.endswith('_best_model.pkl'):
                model_name = filename.replace('_best_model.pkl', '')
                model_path = os.path.join(self.model_dir, filename)
                try:
                    with open(model_path, 'rb') as file:
                        models[model_name] = pickle.load(file)
                except Exception as e:
                    print(f"Error loading model {filename}: {str(e)}")
        
        if not models:
            raise ValueError("No valid models found in the model directory")
        return models
    
    def validate_input_data(self, input_data):
        """
        Validate that all required features are present in the input data.
        This helps catch missing or incorrectly named features early.
        """
        required_inputs = {
            'Power (W)', 'Mass Flowrate (g/min)', 'Travel Velocity (mm/min)',
            'Height (mm)', 'Contact Angle (deg)', 'Spot Size (mm)', 'Width (mm)',
            'powder_material', 'substrate_material'
        }
        
        if isinstance(input_data, dict):
            missing_features = required_inputs - set(input_data.keys())
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
    
    def predict(self, input_data, model_name='random_forest'):
        """
        Make predictions using the specified model with proper error handling
        and data validation.
        
        Parameters:
        -----------
        input_data : dict or DataFrame
            Input data containing process parameters and materials
        model_name : str
            Name of the model to use for prediction
            
        Returns:
        --------
        float
            Predicted aspect ratio
        dict
            Additional information about the prediction including uncertainty estimates
        """
        # Validate inputs
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.available_models.keys())}")
        
        self.validate_input_data(input_data)
        print("Input Data Validated")
        
        try:
            # Process and transform the input data
            processed_data, transformed_df  = self.preprocessing_pipeline.transform(input_data)
            
            # Verify processed data has correct features and order
            expected_features = self.numeric_features + self.material_features
            if not all(col in processed_data.columns for col in expected_features):
                raise ValueError("Processed data missing expected features")
            
            # Ensure features are in the correct order
            processed_data = processed_data[expected_features]
            
            # Make prediction
            prediction = self.available_models[model_name].predict(processed_data)
            
            # Prepare detailed prediction information
            prediction_info = {
                'model_used': model_name,
                'Final_Transformed_features': transformed_df
            }
            
            return prediction[0], prediction_info
            
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
        

input_parameters = {
    'Power (W)': 200,
    'Mass Flowrate (g/min)': 2.5,
    'Travel Velocity (mm/min)': 800,
    'Height (mm)': 10,
    'Contact Angle (deg)': 45,
    'Spot Size (mm)': 0.5,
    'Width (mm)': 2.0,
    'powder_material': 'Steel_based',
    'substrate_material': 'Steel_based'
}

pipeline = AspectRatioPredictionPipeline()
prediction, info = pipeline.predict(input_parameters, 'xgboost')
print(f"Predicted Aspect Ratio: {prediction:.4f}")
print("\nPrediction Details:")
for key, value in info.items():
    print(f"{key}: {value}")
    
