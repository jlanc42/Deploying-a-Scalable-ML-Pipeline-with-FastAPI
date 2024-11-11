import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# TODO: load the census.csv data
project_path = "/Users/johnlancaster/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross-validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=cat_features,
    label="salary",
    training=True  # Indicates that this is training data
)

X_test, y_test, _, _ = process_data(
    X=test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

print(f"Model saved to {model_path}")
print(f"Encoder saved to {encoder_path}")

# Load the model (for testing purposes)
model = load_model(model_path)

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# Iterate through the categorical features
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        # Iterate through the unique values in one categorical feature
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]
            p_slice, r_slice, fb_slice = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slice_value,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            f.write(f"{col}: {slice_value}, Count: {count:,}\n")
            f.write(f"Precision: {p_slice:.4f} | Recall: {r_slice:.4f} | F1: {fb_slice:.4f}\n\n")

print("Model slice performance saved to slice_output.txt")