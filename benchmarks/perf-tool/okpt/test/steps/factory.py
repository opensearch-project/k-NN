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
"""Factory for creating steps."""

from okpt.io.config.parsers.base import ConfigurationError
from okpt.test.steps.base import Step, StepConfig

from okpt.test.steps.steps import CreateIndexStep, DisableRefreshStep, RefreshIndexStep, DeleteIndexStep, \
    TrainModelStep, DeleteModelStep, ForceMergeStep, IngestStep, QueryStep


def create_step(step_config: StepConfig) -> Step:
    if step_config.step_name == CreateIndexStep.label:
        return CreateIndexStep(step_config)
    elif step_config.step_name == DisableRefreshStep.label:
        return DisableRefreshStep(step_config)
    elif step_config.step_name == RefreshIndexStep.label:
        return RefreshIndexStep(step_config)
    elif step_config.step_name == TrainModelStep.label:
        return TrainModelStep(step_config)
    elif step_config.step_name == DeleteModelStep.label:
        return DeleteModelStep(step_config)
    elif step_config.step_name == DeleteIndexStep.label:
        return DeleteIndexStep(step_config)
    elif step_config.step_name == IngestStep.label:
        return IngestStep(step_config)
    elif step_config.step_name == QueryStep.label:
        return QueryStep(step_config)
    elif step_config.step_name == ForceMergeStep.label:
        return ForceMergeStep(step_config)

    raise ConfigurationError("Invalid step {}".format(step_config.step_name))
