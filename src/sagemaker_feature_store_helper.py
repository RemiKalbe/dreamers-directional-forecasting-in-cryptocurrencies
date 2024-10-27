import time

import pandas as pd
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup


def load_feature_store_data(bucket_name: str, feature_group_name: str):
    """Load and prepare data from Feature Store"""
    session = sagemaker.Session()
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=session)

    # Query feature store
    query = f"""
    SELECT *
    FROM "{feature_group.name}"
    ORDER BY EventTime
    """

    # Start the Athena query
    athena = boto3.client("athena")

    # Run the query
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": feature_group.athena_query().database},
        ResultConfiguration={"OutputLocation": f"s3://{bucket_name}/query_results/"},
    )

    query_execution_id = response["QueryExecutionId"]

    # Wait for query to complete
    while True:
        response = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = response["QueryExecution"]["Status"]["State"]

        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break

        time.sleep(5)  # Wait 5 seconds before checking again

    if state == "SUCCEEDED":
        # Get the results
        response = athena.get_query_results(QueryExecutionId=query_execution_id)

        # Convert to pandas DataFrame
        columns = [
            col["Label"]
            for col in response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        ]
        rows = []
        for row in response["ResultSet"]["Rows"][1:]:  # Skip header row
            rows.append([field.get("VarCharValue", "") for field in row["Data"]])

        df = pd.DataFrame(rows, columns=columns)

        # Prepare features and targets
        targets = df["target"]
        features = df.drop(["target", "EventTime", "RecordIdentifier"], axis=1)

        # Split into train/test
        train_size = int(0.8 * len(features))
        train_features = features[:train_size].values
        test_features = features[train_size:].values
        train_targets = targets[:train_size].values
        test_targets = targets[train_size:].values

        return {
            "train_features": train_features,
            "test_features": test_features,
            "train_targets": train_targets,
            "test_targets": test_targets,
        }
    else:
        raise Exception(f"Query failed with state: {state}")
