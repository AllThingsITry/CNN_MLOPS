# pipelines/cnn_pipeline.py
import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.steps import CacheConfig

# ── Session & config ─────────────────────────────────────────────────────────
region     = os.environ.get("AWS_REGION", boto3.session.Session().region_name)
account_id = os.environ.get("AWS_ACCOUNT_ID", boto3.client("sts").get_caller_identity()["Account"])
session    = sagemaker.Session()
role       = os.environ.get("SAGEMAKER_ROLE",
                 f"arn:aws:iam::{account_id}:role/GitHubActions-CNN-Build")

bucket   = f"cnn-mlops-data-{account_id}"
prefix   = "cnn-mlops"
data_uri = f"s3://{bucket}/{prefix}/data/raw"

train_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/cnn-training:latest"
infer_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/cnn-inference:latest"

# Disable cache so preprocessing always re-runs
cache_config = CacheConfig(enable_caching=False)

# ── Step 1: Preprocess ────────────────────────────────────────────────────────
preprocessor = ScriptProcessor(
    image_uri=train_image_uri,
    command=["python3"],
    instance_type="ml.t3.medium",   # valid for processing jobs
    instance_count=1,
    base_job_name="cnn-preprocess",
    role=role,
    sagemaker_session=session,
)

preprocessing_step = ProcessingStep(
    name="PreprocessCNNData",
    processor=preprocessor,
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/{prefix}/data/raw/train",
            destination="/opt/ml/processing/input/train",
        ),
        ProcessingInput(
            source=f"s3://{bucket}/{prefix}/data/raw/test",
            destination="/opt/ml/processing/input/test",
        ), 
    ],
    cache_config=cache_config,
    outputs=[
        ProcessingOutput(output_name="train",      source="/opt/ml/processing/output/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
        ProcessingOutput(output_name="test",       source="/opt/ml/processing/output/test"),
    ],
    code="src/preprocess.py",
)

# ── Step 2: Train ─────────────────────────────────────────────────────────────
estimator = Estimator(
    image_uri=train_image_uri,
    instance_type="ml.m4.xlarge",   # valid for training jobs
    instance_count=1,
    base_job_name="cnn-training",
    role=role,
    sagemaker_session=session,
    hyperparameters={
        "epochs":        20,
        "learning-rate": 0.001,
        "batch-size":    32,
        "num-classes":   10,
    },
    output_path=f"s3://{bucket}/{prefix}/models",
)

training_step = TrainingStep(
    name="TrainCNNModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="application/x-npy",
        ),
        "validation": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="application/x-npy",
        ),
    },
)

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
evaluator = ScriptProcessor(
    image_uri=train_image_uri,
    command=["python3"],
    instance_type="ml.t3.medium",   # valid for processing jobs
    instance_count=1,
    base_job_name="cnn-evaluate",
    role=role,
    sagemaker_session=session,
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation_report.json",
)

evaluation_step = ProcessingStep(
    name="EvaluateCNNModel",
    processor=evaluator,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
    ],
    code="src/evaluate.py",
    property_files=[evaluation_report],
)

# ── Step 4: Register model ────────────────────────────────────────────────────
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation_report.json".format(
            evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json",
    )
)

register_step = RegisterModel(
    name="RegisterCNNModel",
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.t2.medium"],
    transform_instances=["ml.m4.xlarge"],
    model_package_group_name="CNNModelPackageGroup",
    approval_status="PendingManualApproval",
    model_metrics=model_metrics,
)

# ── Step 5: Accuracy gate ─────────────────────────────────────────────────────
condition_step = ConditionStep(
    name="AccuracyGate",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.accuracy.value",
            ),
            right=0.80,
        )
    ],
    if_steps=[register_step],
    else_steps=[],
)

# ── Pipeline definition ───────────────────────────────────────────────────────
pipeline = Pipeline(
    name="cnn-training-pipeline",
    steps=[preprocessing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=session,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    execution.wait()
    print("Pipeline execution complete.")# fix sagemaker trust
# docker train image rebuild
# docker train image rebuild
# docker train image rebuild
# fix sagemaker-training
# fix sagemaker-training
# fix sagemaker-training
# fix sagemaker-training
# fix sagemaker-training
# fix sagemaker-training
