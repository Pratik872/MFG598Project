import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pipeline import AspectRatioPredictionPipeline  # Import your pipeline class

def create_prediction_app():
    st.title("Aspect Ratio Prediction for Direct Energy Deposition")
    
    # Create sidebar for model selection
    st.sidebar.header("Model Selection")
    model_options = ['random_forest', 'xgboost']
    selected_model = st.sidebar.selectbox("Choose a Model", model_options)
    
    # Main input form
    st.header("Input Parameters")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Process Parameters")
        power = st.slider("Power (W)", 
                         min_value=150.0, 
                         max_value=5700.0, 
                         value=200.0, 
                         step=10.0)
        
        mass_flowrate = st.slider("Mass Flowrate (g/min)", 
                                min_value=1.5, 
                                max_value=70.0, 
                                value=2.5, 
                                step=0.1)
        
        travel_velocity = st.slider("Travel Velocity (mm/min)", 
                                  min_value=100.0, 
                                  max_value=1200.0, 
                                  value=800.0, 
                                  step=10.0)
        
        height = st.slider("Height (mm)", 
                         min_value=0.0, 
                         max_value=3.3, 
                         value=10.0, 
                         step=0.5)
        
        contact_angle = st.slider("Contact Angle (deg)", 
                                min_value=0.0, 
                                max_value=168.0, 
                                value=45.0, 
                                step=1.0)
        
    with col2:
        st.subheader("Geometric Parameters")
        spot_size = st.slider("Spot Size (mm)", 
                            min_value=0.5, 
                            max_value=6.0, 
                            value=0.5, 
                            step=0.1)
        
        width = st.slider("Width (mm)", 
                         min_value=0.0, 
                         max_value=6.0, 
                         value=2.0, 
                         step=0.1)
        
        st.subheader("Material Selection")
        powder_material = st.selectbox(
            "Powder Material",
            ["Steel_based", "Cobalt_based", "Copper_based", "Nickel_based", "Titanium_based"]
        )
        substrate_material = st.selectbox(
            "Substrate Material",
            ["Steel_based", "Nickel_based", "Titanium_based"]
        )
    
    # Create prediction button
    if st.button("Predict Aspect Ratio"):
        # Prepare input data dictionary
        input_data = {
            'Power (W)': power,
            'Mass Flowrate (g/min)': mass_flowrate,
            'Travel Velocity (mm/min)': travel_velocity,
            'Height (mm)': height,
            'Contact Angle (deg)': contact_angle,
            'Spot Size (mm)': spot_size,
            'Width (mm)': width,
            'powder_material': powder_material,
            'substrate_material': substrate_material
        }
        
        try:
            # Initialize pipeline and make prediction
            pipeline = AspectRatioPredictionPipeline()
            prediction, info = pipeline.predict(input_data, selected_model)
            
            # Display results
            st.success(f"Predicted Aspect Ratio: {prediction:.4f}")
            
            # Display additional information in an expander
            with st.expander("See Prediction Details"):
                st.write("Model Used:", info['model_used'])
                st.write("Final Transformed Features", info['Final_Transformed_features'])
            
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    create_prediction_app()