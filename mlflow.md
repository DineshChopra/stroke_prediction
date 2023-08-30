## Interacting with MLflow:

MLflow has different interfaces, each with their pros and cons. We introduce  the core functionalities of MLflow through the UI.

### MLflow UI:

To run the MLflow UI locally we use the command:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```


The backend storage is essential to access the features of MLflow, in this command we use a SQLite backend with the file `mlflow.db` in the current running repository. This URI is also given later to the MLflow Python API
`mlflow.set_tracking_uri`.

By accessing the provided local url we can access the UI. Within this UI we have access to MLflow features.

In addition to the backend URI, we can also add an artifact root directory where we store the artifacts for runs, this is done by adding a `--default-artifact-root` paramater:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```