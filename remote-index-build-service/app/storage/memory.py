#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Any
from datetime import datetime, timedelta, timezone
import threading
import time

from models.job import Job
from storage.base import RequestStore
from core.config import Settings

class InMemoryRequestStore(RequestStore):
    def __init__(self, settings: Settings):
        self._store: Dict[str, tuple[Job, datetime]] = {}
        self._lock = threading.Lock()
        self._max_size = settings.request_store_max_size
        self._ttl_seconds = settings.request_store_ttl_seconds

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def add(self, job_id: str, job: Job) -> bool:
        with self._lock:
            if len(self._store) >= self._max_size:
                return False

            self._store[job_id] = (job, datetime.now(timezone.utc))
            return True

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            if job_id in self._store:
                job, timestamp = self._store[job_id]
                if datetime.now(timezone.utc) - timestamp < timedelta(seconds=self._ttl_seconds):
                    return job
                else:
                    del self._store[job_id]
            return None

    def update(self, job_id: str, data: Dict[str, Any]) -> bool:
        with self._lock:
            if job_id not in self._store:
                return False

            job, timestamp = self._store[job_id]
            for key, value in data.items():
                setattr(job, key, value)
            self._store[job_id] = (job, timestamp)
            return True

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._store:
                del self._store[job_id]
                return True
            return False

    def cleanup_expired(self) -> None:
        with self._lock:
            current_time = datetime.now(timezone.utc)
            expiration_threshold = current_time - timedelta(seconds=self._ttl_seconds)

            self._store = {
                job_id: data
                for job_id, data in self._store.items()
                if data[1] > expiration_threshold
            }

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(5)  # Run cleanup every 5 seconds
            self.cleanup_expired()