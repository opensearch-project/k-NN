#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

from pydantic_settings import BaseSettings
from storage.types import RequestStoreType

class Settings(BaseSettings):

    """
    Settings class for the application. Pulls the settings
    from the Docker container environment variables
    """

    # Request Store settings
    request_store_type: RequestStoreType
    request_store_max_size: int
    request_store_ttl_seconds: int

    # Resource Manager settings
    gpu_memory_limit: float
    cpu_memory_limit: float

    # Workflow Executor settings
    max_workers: int

    # Service settings
    service_name: str
    log_level: str

