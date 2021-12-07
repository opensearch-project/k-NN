# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Provides function for reading from files.

Functions:
    get_file_obj(): Get a readable file object.
    parse_yaml(): Parse YAML file from file object.
    parse_yaml_from_path(): Parse YAML file from file path.
    parse_json(): Parse JSON file from file object.
    parse_json_from_path(): Parse JSON file from file path.
"""

import json
from io import TextIOWrapper
from typing import Any, Dict

import yaml

from okpt.io.utils import reader


def get_file_obj(path: str) -> TextIOWrapper:
    """Given a file path, get a readable file object.

    Args:
        file path

    Returns:
        Writeable file object
    """
    return open(path, 'r')


def parse_yaml(file: TextIOWrapper) -> Dict[str, Any]:
    """Parses YAML file from file object.

    Args:
        file: file object to parse

    Returns:
        A dict representing the YAML file.
    """
    return yaml.load(file, Loader=yaml.SafeLoader)


def parse_yaml_from_path(path: str) -> Dict[str, Any]:
    """Parses YAML file from file path.

    Args:
        path: file path to parse

    Returns:
        A dict representing the YAML file.
    """
    file = reader.get_file_obj(path)
    return parse_yaml(file)


def parse_json(file: TextIOWrapper) -> Dict[str, Any]:
    """Parses JSON file from file object.

    Args:
        file: file object to parse

    Returns:
        A dict representing the JSON file.
    """
    return json.load(file)


def parse_json_from_path(path: str) -> Dict[str, Any]:
    """Parses JSON file from file path.

    Args:
        path: file path to parse

    Returns:
        A dict representing the JSON file.
    """
    file = reader.get_file_obj(path)
    return json.load(file)
