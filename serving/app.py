"""

If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
# importing relevant packages 
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import wandb

# Prevent WandB from printing login prompts
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_API_KEY"] = "50c6486ca894b323e5061c4513b477036ac87f8a"   

# Defining LOG File 
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

# Globals and generalized version which supports WANB the model
# Defining WANDB Entity and Project 
WANDB_ENTITY = 'IFT6758-2025-B08'
WANDB_PROJECT = 'IFT6758-Milestone2'

# Correct project path format for WandB API
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

# Defining Model_DIR
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Defining the api, active model state and current model 
api = None
model = None 
current_loaded_model = None 

# Define app
app = Flask(__name__)

# Function to get latest artifact version from wandb
def get_latest_artifact_version(api, model_name):
    """
    Retrieving the latest version of the given model artifact.
    Uses artifact_type() and collections(), fully compatible with all WandB versions.
    """

    # Get the artifact type object
    artifact_type_obj = api.artifact_type("model", PROJECT_PATH)

    # Retrieve all collections under this type
    collections = artifact_type_obj.collections()

    # Find the collection matching the model name
    target_collection = None
    # Iterating through collections
    for c in collections:
        # Checking to see if c.name == model_name
        if c.name == model_name:
            target_collection = c
            break
    # Raise Value Error where you find no collections for model
    if target_collection is None:
        raise ValueError(f"No collection found for model '{model_name}'")

    # Retrieve all versions inside this model's collection
    versions = list(target_collection.artifacts())
    # If no versions, raise Value Error 
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")

    # Sort numerically by version index
    versions.sort(key=lambda a: a.version)

    # Return newest version string, e.g. "logreg_distance:v5"
    return f"{model_name}:{versions[-1].version}"


# Define before_first_request
@app.before_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # Define global model and current_loaded_model, 
    global model, current_loaded_model, api

    # Prevent double initialization
    if getattr(app, "initialized", False):
        return
    # Define app.initialized = True
    app.initialized = True

    # TODO: setup basic logging configuration
    # Setup basic logging once
    logging.basicConfig(
        # Define filename, level, format and force 
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    # Define file_handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    # TODO: any other initialization before the first request (e.g. load default model)
    # Logger info for initalizing the Flask app
    app.logger.info("Initializing Flask app...")

    # Initialize WandB API
    try:
        api = wandb.Api()
    # Catching exception if WANDB authentication failed 
    except Exception as e:
        app.logger.error(f"WANDB authentication failed — running without WANDB. Error: {e}")
        api = None
    # If API is none then define model and current_loaded_model is None 
    if api is None:
        app.logger.error("WANDB unavailable — cannot load default model.")
        model = None
        current_loaded_model = None
        return

    # DEFAULT MODEL NAME 
    def_model = "logreg_distance"
    app.logger.info(f"Preparing to load default model '{def_model}'...")

    try:
        # Dynamically load newest model version
        resolved_version = get_latest_artifact_version(api, def_model)
        app.logger.info(f"Resolved latest version for default model: {resolved_version}")

        # Download artifact
        artifact = api.artifact(f"{PROJECT_PATH}/{resolved_version}")
        downloaded_path = artifact.download(root=str(MODEL_DIR))

        # Find .pkl inside artifact directory
        pkl_files = [f for f in os.listdir(downloaded_path) if f.endswith(".pkl")]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files in downloaded artifact path: {downloaded_path}")
        # Defining model_path, model, and current_loaded_model
        model_path = Path(downloaded_path) / pkl_files[0]
        model = joblib.load(model_path)
        current_loaded_model = def_model

        # Robust feature extraction
        try:
            feature_list = model.feature_names_in_.tolist()
        # Catching exception for default logistic regression models 
        except Exception:
            if def_model == "logreg_distance_angle":
                feature_list = ["distance_from_net", "angle_from_net"]
            elif def_model == "logreg_angle":
                feature_list = ["angle_from_net"]
            else:
                feature_list = ["distance_from_net"]
        # Defining app.logger 
        app.logger.info(
            f"Loaded model '{def_model}' "
            f"(artifact version '{resolved_version}') "
            f"with features: {feature_list}"
        )
    # Catching exception handling for failing to load the default models 
    except Exception as e:
        app.logger.error(f"Failed to load default model: {e}")
        model = None
        current_loaded_model = None
        return


# Defining Logs function
@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    # TODO: read the log file specified and return the data
    # If Log file does not exist write response indicating no logs available 
    if not Path(LOG_FILE).exists():
        response = {"logs": "There are no logs available yet for us to use."}
        return jsonify(response) # Make sure response is josnify
    # Try to open log_file and read its content 
    try:
        with open(LOG_FILE, "r") as file:
            content_of_log = file.read()
        response = {"logs": content_of_log}
        return jsonify(response) # jsonify response 
    # Exception handling to properly read the log file 
    except Exception as e:
        app.logger.error(f"Failed to properly read the log file: {e}") # Failure to properly read the log file 
        response = {"error": str(e)}
        return jsonify(response), 500 # 500 error 


# Define the download_registry_model 
@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    """
    # Get POST json data
    json_payload = request.get_json()
    app.logger.info(json_payload)
    # If model not in json_payload or version return jsonify error 
    if "model" not in json_payload or "version" not in json_payload:
        return jsonify({"error": "model and version fields are required for successful retrieval"}), 400
    # Defining model_requested = json_payload['model'] and version
    model_requested = json_payload["model"]
    version = json_payload["version"]

    global model, current_loaded_model

    # TODO: check to see if the model you are querying for is already downloaded
    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    # If model is already loaded
    
    # If current_loaded_model is model_requested, log the information into log as being already loaded 
    if current_loaded_model == model_requested:
        app.logger.info(f"Requested Model '{model_requested}' is already loaded. No changes made.")
        response = {"status": "already_loaded", "model": model_requested}
        return jsonify(response)
    
    # Define the artifact path 
    artifact_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{model_requested}:{version}"

    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    
    # Write try block
    try:
        # Define artifact and downloaded_path
        artifact = api.artifact(artifact_path)
        downloaded_path = artifact.download(root=str(MODEL_DIR))
        
        # Define pkl files 
        pkl_files = [file for file in os.listdir(downloaded_path) if file.endswith(".pkl")]
        
        # If pkl_files not found, output FileNotFoundError 
        if not pkl_files:
            raise FileNotFoundError(f"There is no pkl file in the downloaded artifact path {downloaded_path}")
        
        # Define model_file and new_model 
        model_file = Path(downloaded_path) / pkl_files[0]
        new_model = joblib.load(model_file)

        # Update model
        model = new_model
        current_loaded_model = model_requested
 
        # Robust feature handling (same as before_first_request)
        try:
            feature_list = new_model.feature_names_in_.tolist()
        except Exception:
            if model_requested == "logreg_distance_angle":
                feature_list = ["distance_from_net", "angle_from_net"]
            elif model_requested == "logreg_angle":
                feature_list = ["angle_from_net"]
            else:
                feature_list = ["distance_from_net"]
        # Logging Active model and features into log 
        app.logger.info(f"ACTIVE MODEL = {current_loaded_model}, FEATURES = {feature_list}")
        app.logger.info(f"Successfully downloaded and loaded model: {current_loaded_model}")
        # Providing successful status message 
        response = {"status": "success", "model": model_requested}
        # Return jsonify response 
        return jsonify(response)
    
    # Exception handling in case there is a failure to download the requested model  
    except Exception as e:
        app.logger.error(f"Failed to download {model_requested}. Error: {e}")
        response = {
            "status": "failed",
            "error": str(e),
            "model_remaining_loaded": current_loaded_model
        }
        return jsonify(response), 500



@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json_payload = request.get_json()
    app.logger.info(json_payload)

    # TODO:
    global model 

    # Need to validate that model is loaded 
    if model is None:
        app.logger.error("No valid model is loaded for predictions to be derived from")
        return jsonify({"error": "No model is currently loaded."}), 500
    
    # Validate JSON Body 
    if json_payload is None:
        app.logger.error("No json payload is received for prediction to be generated.")
        return jsonify({"error": "Request body must contain JSON."}), 400
    
    # Convert JSON to DataFrame 
    try:
        X = pd.DataFrame.from_dict(json_payload)
        app.logger.info(f"Successfully parsed the input into a DataFrame with shape {X.shape}.")
    # Exception Handling 
    except Exception as e:
        app.logger.error(f"Failed to parse json into a DataFrame: {e}")
        return jsonify({"error": "The dataframe format is invalid."}), 400

    try:
        # Case 1: Model includes feature names (ideal)
        expected_cols = None
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            app.logger.info(f"Model reports expected features: {expected_cols}")

            # Verify all required features exist in user input
            missing = [col for col in expected_cols if col not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing required model features: {missing}. "
                    f"Input received: {list(X.columns)}"
                )

            # Filter + reorder columns
            X = X[expected_cols]

        else:
            # Case 2: Model does NOT store feature names → infer feature count
            n_features = model.coef_.shape[1]
            user_cols = list(X.columns)
            # Logging information into the log 
            app.logger.info(
                f"Model has no feature_names_in_. Expecting {n_features} features. "
                f"User provided: {user_cols}"
            )
            # If length of user_cols does not equal number of features raise Value Error 
            if len(user_cols) != n_features:
                raise ValueError(
                    f"Model expects {n_features} features, but received {len(user_cols)}. "
                    f"Input columns: {user_cols}"
                )

            # Deterministic ordering rule: alphabetical
            expected_cols = sorted(user_cols)
            X = X[expected_cols]
            app.logger.info(f"Using inferred feature order: {expected_cols}")
    # Raising exception when feature alignment has failed 
    except Exception as e:
        app.logger.error(f"Feature alignment failed: {e}")
        return jsonify({"error": str(e)}), 500

    # Run Prediction
    try:
        # Probability of class 1 (goal)
        predictions = model.predict_proba(X)[:, 1].tolist()
        # Providing response 
        response = {"predictions": predictions}
        # Successfully logging predictions in app.logger 
        app.logger.info(f"Successfully Generated {len(predictions)} predictions.")
        return jsonify(response)
    # Raising exception if prediction fails. 
    except Exception as e:
        app.logger.error(f"The Prediction has failed because of: {e}")
        return jsonify({"error": str(e)}), 500

