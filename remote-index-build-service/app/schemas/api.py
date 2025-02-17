#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class DataType(str, Enum):
    FLOAT32 = 'fp32'
    FLOAT16 = 'fp16'
    BYTE = 'byte'

class AlgorithmParameters(BaseModel):
    ef_construction: int = 128
    m: int = 16

class IndexParameters(BaseModel):
    engine: str = "faiss"
    name: str = "hnsw"
    space_type: str = "l2"
    algorithm_parameters: AlgorithmParameters = Field(
        default_factory=AlgorithmParameters
    )

class CreateJobRequest(BaseModel):
    repository_type: str
    repository_name: str
    object_path: str
    tenant_id: str
    dimension: int
    doc_count: int
    data_type: DataType = DataType.FLOAT32
    index_parameters: IndexParameters = Field(
        default_factory=IndexParameters
    )

    class Config:
        extra = "forbid"

class CreateJobResponse(BaseModel):
    job_id: str

class GetStatusResponse(BaseModel):
    task_status: str
    knn_index_path: Optional[str] = None
    msg: Optional[str] = None

class CancelJobResponse(BaseModel):
    status: str