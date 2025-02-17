#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel

class RequestParameters(BaseModel):
    object_path: str
    tenant_id: str

    def __str__(self):
        return f"{self.object_path}-{self.tenant_id}"

    def __eq__(self, other):
        if not isinstance(other, RequestParameters):
            return False
        return str(self) == str(other)