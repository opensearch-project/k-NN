#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0
from core.exceptions import HashCollisionError, CapacityError
from fastapi import APIRouter, HTTPException, Request
from schemas.api import CreateJobRequest, CreateJobResponse

import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/_build")
def create_job(create_job_request: CreateJobRequest, request: Request) -> CreateJobResponse:

    logger.info(f"Received create job request: {create_job_request}")

    try:
        job_service = request.app.state.job_service
        job_id = job_service.create_job(create_job_request)
    except HashCollisionError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except CapacityError as e:
        raise HTTPException(status_code=507, detail=str(e))
    return CreateJobResponse(job_id=job_id)