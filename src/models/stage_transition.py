from mlflow.tracking import MlflowClient

client = MlflowClient('http://ec2-13-203-154-81.ap-south-1.compute.amazonaws.com:5000/')

model_name = 'cars'

new_alias = 'champion'

client.set_registered_model_alias(
    name=model_name,
    alias=new_alias,
    version=3
)