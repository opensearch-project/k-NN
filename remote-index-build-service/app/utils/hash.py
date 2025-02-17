#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

import hashlib
from models.job import RequestParameters

def generate_job_id(request_parameters: RequestParameters) -> str:
    combined = str(request_parameters).encode()
    return hashlib.sha256(combined).hexdigest()