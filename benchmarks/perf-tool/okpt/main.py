# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

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
