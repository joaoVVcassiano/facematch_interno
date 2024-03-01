import logging
from pathlib import Path
from typing import Any, Dict, List

from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.model import SKLearnModel

def init_pipeline(
    pipeline: Pipeline, session: PipelineSession, parameters: Dict[str, Any], tags: List[Dict[str, Any]]
) -> ConditionStep:
    """Step Model Register"""
    logging.info("✅ model-register ✅".center(100, "*"))

    step_train = pipeline.steps[1]
    step_eval = pipeline.steps[2]
    model_package_group_name = "facematch-model-v1"

    model = SKLearnModel(
        name=model_package_group_name,
        role=parameters.get("role"),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        framework_version="1.0-1",
        py_version="py3",
        source_dir=str(Path(__file__).resolve()),
        entry_point="inference.py",
        sagemaker_session=session,
    )
    
    register_args = model.register(
        description="FaceMatch V1",
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.r5d.xlarge",
            "ml.r5d.2xlarge",
            "ml.r5d.4xlarge",
            "ml.r5d.12xlarge",
        ],
        transform_instances=[
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
        ],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
        domain="MACHINE_LEARNING",
        task="CLASSIFICATION",
        tags=tags
    )

    step_register = ModelStep(name="CreateModel", step_args=register_args)

    return step_register