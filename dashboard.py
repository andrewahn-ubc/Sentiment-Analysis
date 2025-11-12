import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime

# Configuration
API_URL = "http:/18.119.128.39:8000"

st.set_page_config(page_title="Sentiment Analysis Model Serving Dashboard", layout="wide")

st.title("Sentiment Analysis Model Serving Dashboard")
st.markdown("Real-time monitoring and testing for production ML models")

# Sidebar for testing
st.sidebar.header("Test Predictions")
test_text = st.sidebar.text_area("Enter text to classify:", "I love Lebron James!")
model_choice = st.sidebar.selectbox("Select model:", ["Auto (A/B Test)", "DistilBERT", "RoBERTa"])

if st.sidebar.button("Get Prediction"):
    try:
        payload = {"text": test_text}
        if model_choice != "Auto (A/B Test)":
            payload["model"] = model_choice
        
        response = requests.post(f"{API_URL}/predict", json=payload)
        result = response.json()    
        
        st.sidebar.success(f"**Prediction:** {result['prediction']}")
        st.sidebar.info(f"**Confidence:** {result['confidence']:.2%}")
        st.sidebar.metric("Latency", f"{result['latency']:.2f} ms")
        st.sidebar.caption(f"Model: {result['model_version']}")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Main dashboard - metrics
st.header("Model Performance Metrics")

# Fetch metrics
try:
    response = requests.get(f"{API_URL}/metrics")
    metrics = response.json()
    
    if metrics:
        # Create columns for each model
        cols = st.columns(len(metrics))
        
        for idx, (model_name, data) in enumerate(metrics.items()):
            with cols[idx]:
                st.subheader(f"{model_name}")
                st.metric("Total Requests", data['count'])
                st.metric("Avg Latency", f"{data['avg_latency_ms']:.2f} ms")
                st.metric("Error Rate", f"{data['error_rate']:.2f}%")
    else:
        st.info("No metrics yet. Make some predictions to see data!")

except Exception as e:
    st.info("No metrics yet. Make some predictions to see data!")

# A/B Testing Configuration
st.header("A/B Testing Configuration")
col1, col2 = st.columns(2)

with col1:
    weight_a = st.slider("DistilBERT Weight", 0.0, 1.0, 0.5, 0.05)
with col2:
    weight_b = 1.0 - weight_a
    st.metric("RoBERTa Weight", f"{weight_b:.2f}")

if st.button("Update A/B Weights"):
    try:
        response = requests.post(
            f"{API_URL}/config/weights",
            json={"DistilBERT": weight_a, "RoBERTa": weight_b}
        )
        st.success(f"Weights updated! {response.json()}")
    except Exception as e:
        st.error(f"Failed to update weights: {str(e)}")