# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Provides ToolParser.

Classes:
    ToolParser: Tool config parser.
"""
from dataclasses import dataclass
from io import TextIOWrapper
from typing import List

from okpt.io.config.parsers import base
from okpt.test.steps.base import Step, StepConfig
from okpt.test.steps.factory import create_step


@dataclass
class TestConfig:
    test_name: str
    test_id: str
    endpoint: str
    port: int
    num_runs: int
    show_runs: bool
    setup: List[Step]
    steps: List[Step]
    cleanup: List[Step]


class TestParser(base.BaseParser):
    """Parser for Test config.

    Methods:
        parse: Parse and validate the Test config.
    """

    def __init__(self):
        super().__init__('test')

    def parse(self, file_obj: TextIOWrapper) -> TestConfig:
        """See base class."""
        config_obj = super().parse(file_obj)

        implicit_step_config = dict()
        if 'endpoint' in config_obj:
            implicit_step_config['endpoint'] = config_obj['endpoint']

        if 'port' in config_obj:
            implicit_step_config['port'] = config_obj['port']

        # Each step should have its own parse - take the config object and check if its valid
        setup = []
        if 'setup' in config_obj:
            setup = [create_step(StepConfig(step["name"], step, implicit_step_config)) for step in config_obj['setup']]

        steps = [create_step(StepConfig(step["name"], step, implicit_step_config)) for step in config_obj['steps']]

        cleanup = []
        if 'cleanup' in config_obj:
            cleanup = [create_step(StepConfig(step["name"], step, implicit_step_config)) for step
                       in config_obj['cleanup']]

        test_config = TestConfig(
            endpoint=config_obj['endpoint'],
            port=config_obj['port'],
            test_name=config_obj['test_name'],
            test_id=config_obj['test_id'],
            num_runs=config_obj['num_runs'],
            show_runs=config_obj['show_runs'],
            setup=setup,
            steps=steps,
            cleanup=cleanup
        )

        return test_config
