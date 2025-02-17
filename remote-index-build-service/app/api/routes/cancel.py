#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0
from fastapi import APIRouter, HTTPException, Request
from schemas.api import CancelJobResponse

router = APIRouter()

@router.post("/_cancel/{job_id}")
def cancel_job(job_id: str, request: Request):

    job_service = request.app.state.job_service
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_service.cancel_job():
        return CancelJobResponse(status="success")
    else:
        return CancelJobResponse(status="fail")
