import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ 
    Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier or other trained model.
        Trained machine learning model.
    X : np.array
        Data used for prediction.
        
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """ 
    Serializes model to a file.

    Inputs
    ------
    model : RandomForestClassifier or any sklearn model.
        Trained machine learning model or OneHotEncoder.
        
    path : str
        Path to save pickle file.
        
    """
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    """ 
    Loads pickle file from `path` and returns it.

    Inputs
    ------
    
    path : str
        Path to load pickle file from.

    
    Returns
   -------
   object:
       Loaded object (model or encoder).
    
   """
    with open(path, 'rb') as file:
       return pickle.load(file)


def performance_on_categorical_slice(
        data, column_name, slice_value, categorical_features,
        label, encoder, lb, model):
    
   """ 
   Computes the model metrics on a slice of the data specified by a column name and value.

   Processes the data using one hot encoding for the categorical features and a label binarizer for the labels. 
   This can be used in either training or inference/validation.

   Inputs:
   -------
   data : pd.DataFrame
       Dataframe containing the features and label. Columns in `categorical_features`
       
   column_name : str 
       Column containing the sliced feature.
       
   slice_value : str/int/float 
       Value of the slice feature.
       
   categorical_features: list 
       List containing the names of the categorical features (default=[]).
       
   label : str 
       Name of the label column in `X`. If None, then an empty array will be returned for y (default=None).
       
   encoder : sklearn.preprocessing._encoders.OneHotEncoder 
       Trained sklearn OneHotEncoder used if training=False.
       
   lb : sklearn.preprocessing._label.LabelBinarizer 
       Trained sklearn LabelBinarizer used if training=False.
       
   model: RandomForestClassifier or other trained classifier 
       Model used for prediction.

   Returns:
   -------
   precision : float 
   
   recall: float
   
   fbeta: float

   """

   # Filter data by slice value in specified column name
   sliced_data = data[data[column_name] == slice_value]

   # Process data using process_data function (training=False)
   X_slice, y_slice, _, _ = process_data(
       sliced_data,
       categorical_features=categorical_features,
       label=label,
       encoder=encoder,
       lb=lb,
       training=False)

   # Make predictions on sliced data using inference function
   preds = inference(model, X_slice)

   # Compute metrics using compute_model_metrics function
   precision, recall, fbeta = compute_model_metrics(y_slice, preds)

   return precision, recall, fbeta
