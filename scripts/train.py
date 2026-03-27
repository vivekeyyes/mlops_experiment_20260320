"""
This script implements a training pipeline integrated with MLflow
for experiment tracking, reproducibility, and artifact management.


NOTES:
# Use unique run_name for each training run
# Experiment name can be same for all training runs , artifact root location is immutable for the experiment name
# Backend store uri and artifact location is defined in the script itself, cmd is not required
# MLflow server is started locally on port 5000

"""


import tensorflow as tf
import os
import time

import mlflow
import subprocess
import tempfile
from mlflow.tracking import MlflowClient
import pandas as pd


######### CONFIG ############

# mlflow config
run_name = "20260327 demo run"
experiment_name = "cat vs dog classifier demo training 20260327"

backend_store_uri = "file:///D:/Automation_pipeline/Full_Pipeline/20260327/artifacts/mlruns"
default_artifact_root = "file:///D:/Automation_pipeline/Full_Pipeline/20260327/artifacts/mlflow_artifacts/Training_runs/"+run_name
new_artifact_root = default_artifact_root

data_repo_path = r"D:\Automation_pipeline\Full_Pipeline\Dataset\Train_data" # dvc repo

# training config
dataset_dir = r'D:\Automation_pipeline\Full_Pipeline\Dataset\sample_cata_dog\train'
batch_size = 16
img_size = (160, 160)
epochs = 4
model_save_dir = 'saved_models'

#--------------------------------------------------------

###### MLFLOW SETUP ##########

# End any active MLflow run
if mlflow.active_run():
    mlflow.end_run()

# Start the MLflow server programmatically
mlflow_server_command = [
    "mlflow", "server",
    "--backend-store-uri", backend_store_uri,
    "--default-artifact-root", default_artifact_root,
    "--host", "0.0.0.0",
    "--port", "5000"
]

print("Starting the MLflow server...")
mlflow_server = subprocess.Popen(mlflow_server_command)

time.sleep(5)  # Wait for the server to initialize


try:
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    client = MlflowClient()

    # Define experiment details
    exp_name = experiment_name
    #new_artifact_root = "file:///D:/Automation_pipeline/Full_Pipeline/20260311/artifacts/mlflow_artifacts/Training_runs/"+run_name

    # Check if the experiment exists
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is not None:
        print(f"Experiment '{exp_name}' exists with artifact location: {experiment.artifact_location}")
    else:
        # If the experiment does not exist, create it
        experiment_id = client.create_experiment(name=exp_name, artifact_location=new_artifact_root)
        print(f"Created experiment '{exp_name}' with ID: {experiment_id}")

    # Set the experiment for new runs
    mlflow.set_experiment(exp_name)

    # Override the artifact root for the new run
    os.environ["MLFLOW_ARTIFACT_URI"] = new_artifact_root

    # Start a new run
    run = mlflow.start_run(run_name=run_name)
    print(f"Started run with ID: {run.info.run_id}")

    # Log a sample artifact to test
    #with open("test_artifact.txt", "w") as f:
    #    f.write("This is a test artifact.")
    #mlflow.log_artifact("test_artifact.txt")

except Exception as e:
    print(f"Error occurred: {e}")

#---------------------------------------------------------

#### CODE and DATA version tracking using commit hash
def get_git_commit(repo_path):
    return subprocess.check_output(
        ["git", "-C", repo_path, "rev-parse", "HEAD"]
    ).decode("utf-8").strip()

data_version = get_git_commit(data_repo_path)

mlflow.set_tag("data_repo_commit", data_version)

# code commit
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]
).decode("utf-8").strip()

mlflow.set_tag("git_commit", commit)

##### Training

# === Custom Callback ===

class LogBestModelToMLflow(tf.keras.callbacks.Callback):
    def __init__(self, model_name, classes):
        super().__init__()
        self.model_name = model_name
        self.classes = classes
        self.run_name = run_name
        self.best_val_loss = float("inf")
        self.best_model = None
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Extract metrics
        loss = logs.get("loss")
        acc = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", loss, step=epoch)
        if acc is not None:
            mlflow.log_metric("train_accuracy", acc, step=epoch)
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        if val_acc is not None:
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model = tf.keras.models.clone_model(self.model)
            self.best_model.set_weights(self.model.get_weights())
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        if self.best_model is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

            print(f"Logging best model (val_loss={self.best_val_loss:.4f}) to MLflow...")
            #mlflow.tensorflow.log_model(model=self.best_model, artifact_path = run_name + "/" + "models" + '/' + f"{self.model_name}")

            mlflow.tensorflow.log_model(model=self.best_model,
                                        artifact_path= "models" + '/' + f"{self.model_name}")

            # Log class labels
            class_df = pd.DataFrame({'classes': self.classes})
            class_file = f"class_names_epoch_{self.best_epoch}.csv"
            class_df.to_csv(class_file, index=False)
            mlflow.log_artifact(class_file)
            os.remove(class_file)


# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Save class names BEFORE modifying dataset
class_names = train_ds.class_names

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=img_size + (3,)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#class_names = train_ds.class_names
#class_names = ['cat', 'dog']

log_best_cb = LogBestModelToMLflow(model_name="cat_dog_classifier", classes=class_names)

# Log Parameters (NEW)
print("Logging important parameters to MLflow...")
mlflow.log_params({

    "image_width": img_size[0],
    "image_height": img_size[1],
    "batch_size": batch_size,
    "validation_split": 0.2,
    "classes": ",".join(class_names),
    "epochs": epochs,
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "output_activation": "sigmoid"
})

#mlflow.log_artifact("train.py")
mlflow.log_artifact(os.path.abspath(__file__))

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[log_best_cb]
    )


#### stop mlflow server

mlflow.end_run()
print("Stopping the MLflow server...")
mlflow_server.terminate()
mlflow_server.wait()
print("MLflow server stopped.")