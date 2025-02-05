#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from models.job import Job

class RequestStore(ABC):
    @abstractmethod
    def add(self, job_id: str, job: Job) -> bool:
        """Add a job to the store"""
        pass

    @abstractmethod
    def get(self, job_id: str) -> Optional[Job]:
        """Retrieve a job from the store"""
        pass

    @abstractmethod
    def update(self, job_id: str, data: Dict[str, Any]) -> bool:
        """Update a job in the store"""
        pass

    @abstractmethod
    def delete(self, job_id: str) -> bool:
        """Delete a job from the store"""
        pass

    @abstractmethod
    def cleanup_expired(self) -> None:
        """Clean up expired entries"""
        pass