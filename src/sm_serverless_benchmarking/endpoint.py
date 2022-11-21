import datetime as dt
import logging
import time
import uuid
from dataclasses import dataclass

import boto3
import botocore
from botocore.config import Config

logger = logging.getLogger(__name__)

ALLOWED_MEM_VALUES = {1024, 2048, 3072, 4096, 5120, 6144}
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 200


@dataclass(unsafe_hash=True)
class ServerlessEndpoint:
    model_name: str
    memory_size: int = 1024
    max_concurrency: int = 1
    _created: bool = False
    _deployment_failed: bool = False

    def __post_init__(self):

        self._validate_inital_config()
        self._sm_client = boto3.client("sagemaker")

        # disable retries when throttled
        self._boto_config = Config(retries={"max_attempts": 1, "mode": "standard"})
        self._smr_client = boto3.client("sagemaker-runtime", config=self._boto_config)
        self._cw_client = boto3.client("cloudwatch")
        self._endpoint_name = f"{self.model_name[:50]}-ep-{str(uuid.uuid1())[:5]}"

    def _validate_inital_config(self):
        if self.memory_size not in ALLOWED_MEM_VALUES:
            raise ValueError(
                f"{self.memory_size} is not a valid memory_size value. Valid values are {ALLOWED_MEM_VALUES}"
            )

        if (self.max_concurrency < MIN_CONCURRENCY) | (
            self.max_concurrency > MAX_CONCURRENCY
        ):
            raise ValueError(
                f"max_concurrency must fall within the {MIN_CONCURRENCY} to {MAX_CONCURRENCY} range"
            )

    def _create_endpoint_config(self):
        self._endpoint_config_name = (
            f"{self.model_name[:50]}-cfg-{str(uuid.uuid1())[:5]}"
        )

        endpoint_config_response = self._sm_client.create_endpoint_config(
            EndpointConfigName=self._endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "variant1",
                    "ModelName": self.model_name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": self.memory_size,
                        "MaxConcurrency": self.max_concurrency,
                    },
                }
            ],
        )

        return self._endpoint_config_name

    def create_endpoint(self):

        ep_config = self._create_endpoint_config()
        create_endpoint_response = self._sm_client.create_endpoint(
            EndpointName=self._endpoint_name, EndpointConfigName=ep_config
        )
        self.wait()

        if self._deployment_failed:
            logger.warn(f"Failed to deploy endpoint {self.endpoint_name} with memory_size {self.memory_size}. Failure reason: {self._failure_reason} Endpoint will not be used")
        else:
            self._created = True

    def _validate_deployment(self):
        assert (
            self._created
        ), "Operation can not be performed because the endpoint has not been deployed"

    def describe_endpoint(self):

        resp = self._sm_client.describe_endpoint(EndpointName=self._endpoint_name)

        return resp

    @property
    def endpoint_name(self):
        return self._endpoint_name

    def update_endpoint(self, memory_size, max_concurrency):

        if (memory_size == self.memory_size) & (
            max_concurrency == self.max_concurrency
        ):
            logger.info(
                "Updated configuration matches current configuration. No update required"
            )
            return None

        self._validate_deployment()

        current_config_name = self._endpoint_config_name
        updated_endpoint_config_name = (
            f"{self.model_name[:50]}-cfg-{str(uuid.uuid1())[:5]}"
        )
        updated_config = [
            {
                "VariantName": "variant1",
                "ModelName": self.model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": memory_size,
                    "MaxConcurrency": max_concurrency,
                },
            }
        ]

        endpoint_config_response = self._sm_client.create_endpoint_config(
            EndpointConfigName=updated_endpoint_config_name,
            ProductionVariants=updated_config,
        )

        self._sm_client.update_endpoint(
            EndpointName=self._endpoint_name,
            EndpointConfigName=updated_endpoint_config_name,
        )

        self.wait()
        self._endpoint_config_name = updated_endpoint_config_name
        self._sm_client.delete_endpoint_config(EndpointConfigName=current_config_name)
        self.memory_size = memory_size
        self.max_concurrency = max_concurrency

    def get_endpoint_metric(
        self, namespace: str, metric_name: str, lookback_window: dt.timedelta
    ):

        response = self._cw_client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=[
                {"Name": "EndpointName", "Value": self.endpoint_name},
                {"Name": "VariantName", "Value": "variant1"},
            ],
            StartTime=dt.datetime.utcnow() - lookback_window,
            EndTime=dt.datetime.utcnow(),
            Period=3600 * 24,
            Statistics=["Average", "Minimum", "Maximum"],
            ExtendedStatistics=["p25", "p50", "p75"],
        )
        metric_data_points = response["Datapoints"]

        if len(metric_data_points) == 0:
            logger.warn(f"Did not get any CloudWatch data for the {metric_name} metric for endpoint {self.endpoint_name}")
            return {"metric_name": metric_name}
        else:
            metric = metric_data_points[0]

        extended_statistics = metric.pop("ExtendedStatistics")
        metric.update(extended_statistics)
        metric["metric_name"] = metric_name

        return metric

    def get_endpoint_metrics(self, lookback_window: dt.timedelta):

        all_metrics = []
        memory_util_metric = self.get_endpoint_metric(
            namespace="/aws/sagemaker/Endpoints",
            metric_name="MemoryUtilization",
            lookback_window=lookback_window,
        )
        all_metrics.append(memory_util_metric)

        for metric_name in ["ModelSetupTime", "ModelLatency", "OverheadLatency"]:
            metric = self.get_endpoint_metric(
                namespace="AWS/SageMaker",
                metric_name=metric_name,
                lookback_window=lookback_window,
            )
            all_metrics.append(metric)

        return all_metrics

    def clean_up(self):
        
        self._sm_client.delete_endpoint(EndpointName=self._endpoint_name)
        self._sm_client.delete_endpoint_config(
            EndpointConfigName=self._endpoint_config_name
        )
        self._created = False

    def invoke_endpoint(self, invoke_args):

        self._validate_deployment()

        resp = self._smr_client.invoke_endpoint(
            EndpointName=self._endpoint_name, **invoke_args
        )

        return resp

    def wait(self):

        time.sleep(2)  # wait a few seconds for status to update
        waiter = self._sm_client.get_waiter("endpoint_in_service")
        print(f"Waiting for endpoint {self._endpoint_name} to start...")

        try:
            waiter.wait(EndpointName=self._endpoint_name)

        except botocore.exceptions.WaiterError as err:
            self._deployment_failed = True
            self._failure_reason = self.describe_endpoint().get("FailureReason")
            self.clean_up()
            return None

        resp = self.describe_endpoint()
        print(f"Endpoint {self.endpoint_name} Status: {resp['EndpointStatus']}")

        return resp





