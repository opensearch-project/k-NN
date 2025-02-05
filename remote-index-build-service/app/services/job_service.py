from typing import Optional
from core.exceptions import HashCollisionError, CapacityError
from core.resources import ResourceManager
from executors.workflow_executor import WorkflowExecutor
from models.job import Job, JobStatus
from models.request import RequestParameters
from models.workflow import BuildWorkflow
from utils.hash import generate_job_id
from utils.memory import calculate_memory_requirements
from utils.request import create_request_parameters
from storage.base import RequestStore
from schemas.api import CreateJobRequest

import logging

logger = logging.getLogger(__name__)


class JobService:
    def __init__(
            self,
            request_store: RequestStore,
            workflow_executor: WorkflowExecutor,
            resource_manager: ResourceManager,
            total_gpu_memory: float,
            total_cpu_memory: float
    ):
        self.request_store = request_store
        self.workflow_executor = workflow_executor
        self.total_gpu_memory = total_gpu_memory
        self.total_cpu_memory = total_cpu_memory
        self.resource_manager = resource_manager

    def _validate_job_existence(self, job_id: str, request_parameters: RequestParameters) -> bool:
        job = self.request_store.get(job_id)
        if job:
            if job.compare_request_parameters(request_parameters):
                return True
            raise HashCollisionError(f"Hash collision detected for job_id: {job_id}")
        return False

    def _get_required_resources(self, create_job_request: CreateJobRequest) -> tuple[float, float]:
        gpu_mem, cpu_mem = calculate_memory_requirements(
            create_job_request.dimension,
            create_job_request.doc_count,
            create_job_request.data_type,
            create_job_request.index_parameters.algorithm_parameters.m
        )

        logger.info(f"Job id requirements: GPU memory: {gpu_mem}, CPU memory: {cpu_mem}")
        if not self.resource_manager.can_allocate(gpu_mem, cpu_mem):
            raise CapacityError(f"Insufficient available GPU and CPU resources to process job")

        return gpu_mem, cpu_mem

    def _add_to_request_store(self, job_id: str, request_parameters: RequestParameters) -> None:
        result = self.request_store.add(
            job_id,
            Job(
                id=job_id,
                status=JobStatus.RUNNING,
                request_parameters=request_parameters
            )
        )

        if not result:
            raise CapacityError("Could not add item to request store")

    def _create_workflow(self, job_id: str, gpu_mem: float, cpu_mem: float, create_job_request: CreateJobRequest) -> BuildWorkflow:
        workflow = BuildWorkflow(
            job_id=job_id,
            gpu_memory_required=gpu_mem,
            cpu_memory_required=cpu_mem,
            create_job_request=create_job_request
        )

        # Allocate resources
        allocation_success = self.resource_manager.allocate(
            workflow.gpu_memory_required,
            workflow.cpu_memory_required
        )

        if not allocation_success:
            self.request_store.delete(job_id)
            raise CapacityError(
                f"Insufficient available resources to process workflow {workflow.job_id}"
            )

        return workflow

    def create_job(self, create_job_request: CreateJobRequest) -> str:
        """
        Creates a new job based on the provided request.

        Args:
            create_job_request: The job creation request containing necessary parameters

        Returns:
            str: The ID of the created job

        Raises:
            HashCollisionError: If same job id for different request exists
            CapacityError: If worker does not have memory for request
        """
        # Create parameters and validate job
        request_parameters = create_request_parameters(create_job_request)
        job_id = generate_job_id(request_parameters)
        job_exists = self._validate_job_existence(job_id, request_parameters)
        if job_exists:
            logger.info(f"Job with id {job_id} already exists")
            return job_id

        gpu_mem, cpu_mem = self._get_required_resources(create_job_request)

        self._add_to_request_store(job_id, request_parameters)
        logger.info(f"Added job to request store with job id: {job_id}")

        # used to determine if clean up is necessary
        workflow = None
        submit_success = False

        try:
            workflow = self._create_workflow(job_id, gpu_mem, cpu_mem, create_job_request)
            logger.info(
                f"Worker resource status - GPU: {self.resource_manager.get_available_gpu_memory():,} units, "
                f"CPU: {self.resource_manager.get_available_cpu_memory():,} units"
            )
            submit_success = self.workflow_executor.submit_workflow(workflow)
            logger.info(f"Successfully created workflow with job id: {job_id}")

        finally:
            if not submit_success and workflow:
                # submitting to the thread pool executor failed,
                # so we need to clean up the resources and request store
                self.resource_manager.release(
                    workflow.gpu_memory_required,
                    workflow.cpu_memory_required
                )
                self.request_store.delete(workflow.job_id)

        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.request_store.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        return self.request_store.delete(job_id)

