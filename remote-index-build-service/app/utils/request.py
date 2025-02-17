#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

from schemas.api import CreateJobRequest
from models.request import RequestParameters

def create_request_parameters(create_job_request: CreateJobRequest) -> RequestParameters:
    return RequestParameters(
        object_path=create_job_request.object_path,
        tenant_id=create_job_request.tenant_id
    )