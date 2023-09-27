# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
"""Factory for creating steps."""

from okpt.io.config.parsers.base import ConfigurationError
from okpt.test.steps.base import Step, StepConfig

from okpt.test.steps.steps import CreateIndexStep, DisableRefreshStep, RefreshIndexStep, DeleteIndexStep, \
    TrainModelStep, DeleteModelStep, ForceMergeStep, ClearCacheStep, IngestStep, IngestMultiFieldStep, \
    QueryStep, QueryWithFilterStep, GetStatsStep, WarmupStep


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
    elif step_config.step_name == IngestMultiFieldStep.label:
        return IngestMultiFieldStep(step_config)
    elif step_config.step_name == QueryStep.label:
        return QueryStep(step_config)
    elif step_config.step_name == QueryWithFilterStep.label:
        return QueryWithFilterStep(step_config)
    elif step_config.step_name == ForceMergeStep.label:
        return ForceMergeStep(step_config)
    elif step_config.step_name == ClearCacheStep.label:
        return ClearCacheStep(step_config)
    elif step_config.step_name == GetStatsStep.label:
        return GetStatsStep(step_config)
    elif step_config.step_name == WarmupStep.label:
        return WarmupStep(step_config)

    raise ConfigurationError(f'Invalid step {step_config.step_name}')
