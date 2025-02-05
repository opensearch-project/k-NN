from api.routes import build, status, cancel
from fastapi import FastAPI
from core.config import Settings
from core.resources import ResourceManager
from executors.workflow_executor import WorkflowExecutor
from models.workflow import BuildWorkflow
from services.index_builder import IndexBuilder
from services.job_service import JobService
from storage.factory import RequestStoreFactory
from utils.logging_config import configure_logging

import logging

settings = Settings()

configure_logging(settings.log_level)

logger = logging.getLogger(__name__)

request_store = RequestStoreFactory.create(
    store_type=settings.request_store_type,
    settings=settings
)

resource_manager = ResourceManager(
    total_gpu_memory=settings.gpu_memory_limit,
    total_cpu_memory=settings.cpu_memory_limit
)

index_builder = IndexBuilder(settings)

workflow_executor = WorkflowExecutor(
    max_workers=settings.max_workers,
    request_store=request_store,
    resource_manager=resource_manager,
    build_index_fn=index_builder.build_index
)

job_service = JobService(
    request_store=request_store,
    resource_manager=resource_manager,
    workflow_executor=workflow_executor,
    total_gpu_memory=settings.gpu_memory_limit,
    total_cpu_memory=settings.cpu_memory_limit
)

app = FastAPI(
    title=settings.service_name
)

app.state.job_service = job_service

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application ...")
    workflow_executor.shutdown()

app.include_router(build.router)
app.include_router(status.router)
app.include_router(cancel.router)