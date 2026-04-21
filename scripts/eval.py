"""
This script implements a model evaluation pipeline integrated with MLflow
for experiment tracking, reproducibility, and artifact management.


NOTES:
# Run_name should be same as corresponding training run
# Experiment name can be same for all eval runs , artifact root location is immutable for the experiment name
# Backend store uri and artifact location is defined in the script itself, cmd is not required
# MLflow server is started locally on port 5000

# Best model from corresponding training is automatically loaded using run_name, Load the best model from artifact folder

"""

import os
import io
import tempfile
import itertools
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import subprocess
import tempfile
from mlflow.tracking import MlflowClient
import pandas as pd
import time

import warnings
warnings.filterwarnings("ignore", message="Corrupt JPEG data")


######### CONFIG ############

# mlflow config
run_name = "20260421 demo run"
experiment_name = "demo eval 20260421"

backend_store_uri = "file:///D:/Automation_pipeline/Full_Pipeline/20260421/artifacts/mlruns_eval/mlruns"
default_artifact_root = "file:///D:/Automation_pipeline/Full_Pipeline/20260421/artifacts/mlflow_artifacts/Eval_runs/"+run_name
new_artifact_root = default_artifact_root

base_artifact_path = "D:/Automation_pipeline/Full_Pipeline/20260421/artifacts/mlflow_artifacts/Training_runs/" + run_name

data_repo_path = r"D:\Automation_pipeline\Full_Pipeline\Dataset\Test_data"

# Eval config
eval_dataset_dir = r"D:\Automation_pipeline\Full_Pipeline\Dataset\sample_cata_dog\test"
img_size = (160, 160)
batch_size = 16

# === Deployment thresholds ===
ACCURACY_THRESHOLD = 0.50          # example
LATENCY_THRESHOLD_MS = 200         # example (single image)


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
    #new_artifact_root = "file:///D:/Automation_pipeline/Full_Pipeline/20260311/artifacts/mlflow_artifacts/Eval_runs/"+run_name

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



# === Custom function to log a Matplotlib figure to MLflow ===
def log_figure_to_mlflow2(fig, artifact_path, file_name="figure.png", dpi=300):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi)
    buffer.seek(0)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, file_name)
        with open(temp_path, "wb") as f:
            f.write(buffer.getvalue())
        mlflow.log_artifact(temp_path, artifact_path=artifact_path)

    plt.close(fig)

## code and data version tracking
def get_git_commit(repo_path):
    return subprocess.check_output(
        ["git", "-C", repo_path, "rev-parse", "HEAD"]
    ).decode("utf-8").strip()


data_version = get_git_commit(data_repo_path)

mlflow.set_tag("data_repo_commit", data_version)

# code commit
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

mlflow.set_tag("git_commit", commit)

# === 1. Load the best model from artifact folder ===

run_dirs = [d for d in os.listdir(base_artifact_path) if os.path.isdir(os.path.join(base_artifact_path, d))]
if not run_dirs:
    raise FileNotFoundError(f"No run folders found in {base_artifact_path}")

run_id_folder = run_dirs[0]

print('run_id_folder:', run_id_folder)

mlflow.log_param("training run_id_folder", run_id_folder)

model_path = os.path.join(base_artifact_path, run_id_folder, 'artifacts', 'models', 'cat_dog_classifier', 'data', 'model').replace(os.sep, '/')
print("model_path:", model_path)

model = tf.keras.models.load_model(model_path)
print(f"Loaded model from: {model_path}")

# === 2. Prepare evaluation dataset ===

eval_ds = tf.keras.preprocessing.image_dataset_from_directory(
    eval_dataset_dir,
    validation_split=0.9,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

classes = eval_ds.class_names

eval_ds = eval_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 3. Evaluate the model ===

results = model.evaluate(eval_ds, return_dict=True)
print(" Evaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# === 4. Generate and log confusion matrix ===

# Get true and predicted labels
y_true = []
y_pred = []

for images, labels in eval_ds:
    preds = model.predict(images)
    preds_binary = (preds > 0.5).astype("int32").flatten()
    y_true.extend(labels.numpy())
    y_pred.extend(preds_binary)

cm = confusion_matrix(y_true, y_pred)


# Plot confusion matrix
fig = plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.tight_layout()

# Log confusion matrix to MLflow
log_figure_to_mlflow2(fig, artifact_path="figures", file_name="confusion_matrix.png")

# Optional: log metrics too (if reattaching to a run)
# mlflow.set_experiment("cat dog classifier exp")
# with mlflow.start_run(run_id=run_id_folder):  # Use run_id_folder if appropriate
#     for metric, value in results.items():
#         mlflow.log_metric(f"eval_{metric}", value)

#mlflow.log_artifact("eval.py")
mlflow.log_artifact(os.path.abspath(__file__))

#--------------------------------------------------------

# ==============================
# Test Cases + Summary Logging
# ==============================

test_summary = {}
all_passed = True

mlflow.log_param("ACCURACY_THRESHOLD", ACCURACY_THRESHOLD)
mlflow.log_param("LATENCY_THRESHOLD_MS", LATENCY_THRESHOLD_MS)


# === DEPLOYMENT GATE 1: Accuracy ===

eval_accuracy = results.get("accuracy")
mlflow.log_metric("eval_accuracy", eval_accuracy)

if eval_accuracy < ACCURACY_THRESHOLD:
    test_summary["accuracy_gate"] = f"FAILED ({eval_accuracy:.4f} < {ACCURACY_THRESHOLD})"
    mlflow.log_param("accuracy_gate", "failed")
    all_passed = False
    print(f" Accuracy gate FAILED: {eval_accuracy:.4f}")
else:
    test_summary["accuracy_gate"] = f"PASSED ({eval_accuracy:.4f})"
    mlflow.log_param("accuracy_gate", "passed")
    print(f" Accuracy gate passed: {eval_accuracy:.4f}")


# === DEPLOYMENT GATE 2: Determinism ===

print("Running inference determinism gate")

for images, labels in eval_ds.take(1):
    sample_image = images[0:1]
    break

pred1 = model.predict(sample_image)
pred2 = model.predict(sample_image)

TOLERANCE = 1e-6
max_diff = float(np.max(np.abs(pred1 - pred2)))

mlflow.log_metric("determinism_max_abs_diff", max_diff)

if max_diff > TOLERANCE:
    test_summary["determinism_gate"] = f"FAILED (max diff {max_diff})"
    mlflow.log_param("determinism_gate", "failed")
    all_passed = False
    print(f" Determinism gate FAILED: {max_diff}")
else:
    test_summary["determinism_gate"] = "PASSED"
    mlflow.log_param("determinism_gate", "passed")
    print(" Determinism gate passed")


# === DEPLOYMENT GATE 3: Latency ===

print("Running single-image inference latency gate")

# Warm-up
_ = model.predict(sample_image)

start_time = time.perf_counter()
_ = model.predict(sample_image)
end_time = time.perf_counter()

latency_ms = (end_time - start_time) * 1000
mlflow.log_metric("single_image_latency_ms", latency_ms)

if latency_ms > LATENCY_THRESHOLD_MS:
    test_summary["latency_gate"] = f"FAILED ({latency_ms:.2f} ms > {LATENCY_THRESHOLD_MS})"
    mlflow.log_param("latency_gate", "failed")
    all_passed = False
    print(f" Latency gate FAILED: {latency_ms:.2f} ms")
else:
    test_summary["latency_gate"] = f"PASSED ({latency_ms:.2f} ms)"
    mlflow.log_param("latency_gate", "passed")
    print(f" Latency gate passed: {latency_ms:.2f} ms")


# === FINAL SUMMARY ===

summary_text = "TEST CASE SUMMARY\n"
summary_text += "====================\n\n"

for test, result in test_summary.items():
    summary_text += f"{test}: {result}\n"

if all_passed:
    summary_text += "\nFINAL STATUS: PASSED (READY FOR DEPLOYMENT)\n"
    mlflow.log_param("deployment_ready", "yes")
    print(" MODEL PASSED ALL DEPLOYMENT GATES")
else:
    summary_text += "\nFINAL STATUS: FAILED (NOT READY FOR DEPLOYMENT)\n"
    mlflow.log_param("deployment_ready", "no")
    print(" MODEL FAILED DEPLOYMENT GATES")


# === SAVE + LOG SUMMARY ===

summary_file = "test_summary.txt"

with open(summary_file, "w") as f:
    f.write(summary_text)

mlflow.log_artifact(summary_file, artifact_path="test_results")

os.remove(summary_file)
mlflow.end_run()

print("Stopping the MLflow server...")

mlflow_server.terminate()
mlflow_server.wait()
print("MLflow server stopped.")

