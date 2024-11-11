import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd


def test_train_model():
    """
    Test if the train_model function returns a RandomForestClassifier.
    """
    X_train, y_train, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    
    model = train_model(X_train, y_train)
    
    assert isinstance(model, RandomForestClassifier), f"Expected RandomForestClassifier but got {type(model)}"

def test_inference():
 """
    Test if the inference function returns predictions with correct shape.
    """
    X_train, y_train, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    
    model = train_model(X_train, y_train)
    
    preds = inference(model, X_train)
    
    assert isinstance(preds, np.ndarray), f"Expected np.ndarray but got {type(preds)}"
    assert preds.shape == (X_train.shape[0],), f"Expected shape {(X_train.shape[0],)} but got {preds.shape}"
 

def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns precision/recall/fbeta correctly.
    """
    # Simulate true labels and predicted labels
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert precision == pytest.approx(0.6667, rel=1e-2), f"Expected precision ~0.6667 but got {precision}"
    assert recall == pytest.approx(1.0), f"Expected recall ~1.0 but got {recall}"
    assert fbeta == pytest.approx(0.8), f"Expected F1 score ~0.8 but got {fbeta}"
