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
""" Runner script that serves as the main controller of the testing tool."""

import logging
import sys
from typing import cast

from okpt.diff import diff
from okpt.io import args
from okpt.io.config.parsers import test
from okpt.io.utils import reader, writer
from okpt.test import runner


def main():
    """Main function of entry module."""
    cli_args = args.get_args()
    output = cli_args.output
    if cli_args.log:
        log_level = getattr(logging, cli_args.log.upper())
        logging.basicConfig(level=log_level)

    if cli_args.command == 'test':
        cli_args = cast(args.TestArgs, cli_args)

        # parse config
        parser = test.TestParser()
        test_config = parser.parse(cli_args.config)
        logging.info('Configs are valid.')

        # run tests
        test_runner = runner.TestRunner(test_config=test_config)
        test_result = test_runner.execute()

        # write test results
        logging.debug(
            f'Test Result:\n {writer.write_json(test_result, sys.stdout, pretty=True)}'
        )
        writer.write_json(test_result, output, pretty=True)
    elif cli_args.command == 'diff':
        cli_args = cast(args.DiffArgs, cli_args)

        # parse test results
        base_result = reader.parse_json(cli_args.base_result)
        changed_result = reader.parse_json(cli_args.changed_result)

        # get diff
        diff_result = diff.Diff(base_result, changed_result,
                                cli_args.metadata).diff()
        writer.write_json(data=diff_result, file=output, pretty=True)
