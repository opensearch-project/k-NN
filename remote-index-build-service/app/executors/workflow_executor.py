#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Optional, Dict, Callable
from models.workflow import BuildWorkflow
from core.resources import ResourceManager
from core.exceptions import BuildError, ObjectStoreError
from storage.base import RequestStore
from models.job import JobStatus

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    def __init__(
            self,
            max_workers: int,
            request_store: RequestStore,
            resource_manager: ResourceManager,
            build_index_fn: Callable[[BuildWorkflow], tuple[bool, Optional[str], Optional[str]]]
    ):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._request_store = request_store
        self._resource_manager = resource_manager
        self._build_index_fn = build_index_fn

    def submit_workflow(self, workflow: BuildWorkflow) -> bool:
        """
        Submit a workflow for execution.
        Returns True if submission was successful.
        """

        # Submit the workflow to thread pool
        self._executor.submit(
            self._execute_workflow,
            workflow
        )

        return True

    def _execute_workflow(self, workflow: BuildWorkflow) -> None:
        """
        Execute the workflow and handle results.
        """
        try:
            logger.info(f"Starting execution of job {workflow.job_id}")

            if not self._request_store.get(workflow.job_id):
                logger.info(f"Job {workflow.job_id} was cancelled during submission")
                return

            # Execute the build
            success, index_path, msg = self._build_index_fn(workflow)

            # Check if job still exists before updating status
            if self._request_store.get(workflow.job_id):
                status = JobStatus.COMPLETED if success else JobStatus.FAILED
                self._request_store.update(
                    workflow.job_id,
                    {
                        "status": status,
                        "knn_index_path": index_path,
                        "msg": msg
                    }
                )

                logger.info(
                    f"Job {workflow.job_id} completed with status {status}"
                )
            else:
                logger.info(f"Job {workflow.job_id} was cancelled during execution")

        except (BuildError, ObjectStoreError, MemoryError, RuntimeError) as e:
            logger.error(
                f"Build process failed for job {workflow.job_id}: {str(e)}"
            )
            self._request_store.update(
                workflow.job.id,
                {
                    "status": status,
                    "knn_index_path": index_path,
                    "msg": str(e)
                }
            )
        finally:
            # Release resources
            self._resource_manager.release(
                workflow.gpu_memory_required,
                workflow.cpu_memory_required
            )


    def shutdown(self) -> None:
        """
        Shutdown the executor
        """
        self._executor.shutdown(wait=True)