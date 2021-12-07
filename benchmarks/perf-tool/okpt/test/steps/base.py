# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Provides base Step interface."""

from dataclasses import dataclass
from typing import Any, Dict, List

from okpt.test import profile


@dataclass
class StepConfig:
    step_name: str
    config: Dict[str, object]
    implicit_config: Dict[str, object]


class Step:
    """Test step interface.

    Attributes:
        label: Name of the step.

    Methods:
        execute: Run the step and return a step response with the label and corresponding measures.
    """

    label = 'base_step'

    def __init__(self, step_config: StepConfig):
        self.step_config = step_config

    def _action(self):
        """Step logic/behavior to be executed and profiled."""
        pass

    def _get_measures(self) -> List[str]:
        """Gets the measures for a particular test"""
        pass

    def execute(self) -> List[Dict[str, Any]]:
        """Execute step logic while profiling various measures.

        Returns:
            Dict containing step label and various step measures.
        """
        action = self._action

        # profile the action with measure decorators - add if necessary
        action = getattr(profile, "took")(action)

        result = action()
        if isinstance(result, dict):
            return [{'label': self.label, **result}]

        raise ValueError("Invalid return by a step")
