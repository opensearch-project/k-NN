#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel
from schemas.api import CreateJobRequest

class BuildWorkflow(BaseModel):
    job_id: str
    gpu_memory_required: float
    cpu_memory_required: float
    create_job_request: CreateJobRequest
