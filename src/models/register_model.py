from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri('http://ec2-13-203-154-81.ap-south-1.compute.amazonaws.com:5000/')
client = MlflowClient(tracking_uri='http://ec2-13-203-154-81.ap-south-1.compute.amazonaws.com:5000/')

run_id = 'cfde94cb78604aa4a6778fcfd1a96d71'

model_path = 's3://autonexusmlflow/222357640871536128/cfde94cb78604aa4a6778fcfd1a96d71/artifacts/MLmodel'

model_uri = f"runs:/{run_id}/best_model"

model_name = 'cars'
result = mlflow.register_model(model_uri=model_uri,name=model_name)

client.update_model_version(
    name=model_name,
    version=result.version,
    description='a new model version added via code'
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key='model',
    value='price pridiction model'
)