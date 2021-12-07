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
"""Parses and defines command line arguments for the program.

Defines the subcommands `test` and `diff` and the corresponding
files that are required by each command.

Functions:
    define_args(): Define the command line arguments.
    get_args(): Returns a dictionary of the command line args.
"""

import argparse
import sys
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Union

_read_type = argparse.FileType('r')
_write_type = argparse.FileType('w')


def _add_config(parser, name, **kwargs):
    """"Add configuration file path argument."""
    opts = {
        'type': _read_type,
        'help': 'Path of configuration file.',
        'metavar': 'config_path',
        **kwargs,
    }
    parser.add_argument(name, **opts)


def _add_result(parser, name, **kwargs):
    """"Add results files paths argument."""
    opts = {
        'type': _read_type,
        'help': 'Path of one result file.',
        'metavar': 'result_path',
        **kwargs,
    }
    parser.add_argument(name, **opts)


def _add_results(parser, name, **kwargs):
    """"Add results files paths argument."""
    opts = {
        'nargs': '+',
        'type': _read_type,
        'help': 'Paths of result files.',
        'metavar': 'result_paths',
        **kwargs,
    }
    parser.add_argument(name, **opts)


def _add_output(parser, name, **kwargs):
    """"Add output file path argument."""
    opts = {
        'type': _write_type,
        'help': 'Path of output file.',
        'metavar': 'output_path',
        **kwargs,
    }
    parser.add_argument(name, **opts)


def _add_metadata(parser, name, **kwargs):
    opts = {
        'action': 'store_true',
        **kwargs,
    }
    parser.add_argument(name, **opts)


def _add_test_cmd(subparsers):
    test_parser = subparsers.add_parser('test')
    _add_config(test_parser, 'config')
    _add_output(test_parser, 'output')


def _add_diff_cmd(subparsers):
    diff_parser = subparsers.add_parser('diff')
    _add_metadata(diff_parser, '--metadata')
    _add_result(
        diff_parser,
        'base_result',
        help='Base test result.',
        metavar='base_result'
    )
    _add_result(
        diff_parser,
        'changed_result',
        help='Changed test result.',
        metavar='changed_result'
    )
    _add_output(diff_parser, '--output', default=sys.stdout)


@dataclass
class TestArgs:
    log: str
    command: str
    config: TextIOWrapper
    output: TextIOWrapper


@dataclass
class DiffArgs:
    log: str
    command: str
    metadata: bool
    base_result: TextIOWrapper
    changed_result: TextIOWrapper
    output: TextIOWrapper


def get_args() -> Union[TestArgs, DiffArgs]:
    """Define, parse and return command line args.

    Returns:
        A dict containing the command line args.
    """
    parser = argparse.ArgumentParser(
        description=
        'Run performance tests against the OpenSearch plugin and various ANN libaries.'
    )

    def define_args():
        """Define tool commands."""

        # add log level arg
        parser.add_argument(
            '--log',
            default='info',
            type=str,
            choices=['debug',
                     'info',
                     'warning',
                     'error',
                     'critical'],
            help='Log level of the tool.'
        )

        subparsers = parser.add_subparsers(
            title='commands',
            dest='command',
            help='sub-command help'
        )
        subparsers.required = True

        # add subcommands
        _add_test_cmd(subparsers)
        _add_diff_cmd(subparsers)

    define_args()
    args = parser.parse_args()
    if args.command == 'test':
        return TestArgs(
            log=args.log,
            command=args.command,
            config=args.config,
            output=args.output
        )
    else:
        return DiffArgs(
            log=args.log,
            command=args.command,
            metadata=args.metadata,
            base_result=args.base_result,
            changed_result=args.changed_result,
            output=args.output
        )
