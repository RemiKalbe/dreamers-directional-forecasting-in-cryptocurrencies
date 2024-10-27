import os
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3

SESSION_PROFILE_NAME = "AdministratorAccess-360053549056"
SM_ROLE = "AmazonSageMaker-ExecutionRole-20241027T082974"
SM_FEATURE_GROUP_NAME = "dreamers-directional-forecasting-in-cryptocurrencies"
SM_S3_BUCKET_NAME = "dreamers-directional-forecasting-in-cryptocurrencies"

# Create boto3 session
boto_session = boto3.Session(profile_name=SESSION_PROFILE_NAME)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

hyperparameters = {
    "seq_length": 256,
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 8,
    "dropout": 0.3,
    # Training hyperparameters
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    # Feature Store parameters
    "feature_group_name": SM_FEATURE_GROUP_NAME,
    "feature_bucket_name": SM_S3_BUCKET_NAME,
}

pytorch_estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    dependencies=[os.path.join("src", "requirements.txt")],
    role=SM_ROLE,
    instance_count=1,
    instance_type="ml.p4d.24xlarge",
    framework_version="2.5.0",
    py_version="py312",
    hyperparameters=hyperparameters,
    distribution={"torch_distributed": {"enabled": True}},
    code_location=f"s3://{SM_S3_BUCKET_NAME}/training-code",
    output_path=f"s3://{SM_S3_BUCKET_NAME}/training-output",
    tensorboard_output_config=True,
)

# Start training
pytorch_estimator.fit()
