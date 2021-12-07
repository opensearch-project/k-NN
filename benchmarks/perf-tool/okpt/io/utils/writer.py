# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Provides functions for writing to file.

Functions:
    get_file_obj(): Get a writeable file object.
    write_json(): Writes a python dictionary to a JSON file
"""

import json
from io import TextIOWrapper
from typing import Any, Dict, TextIO, Union


def get_file_obj(path: str) -> TextIOWrapper:
    """Get a writeable file object from a file path.

    Args:
        file path

    Returns:
        Writeable file object
    """
    return open(path, 'w', encoding='UTF-8')


def write_json(data: Dict[str, Any],
               file: Union[TextIOWrapper, TextIO],
               pretty=False):
    """Writes a dictionary to a JSON file.

    Args:
        data: A dict to write to JSON.
        file: Path of output file.
    """
    indent = 2 if pretty else 0
    json.dump(data, file, indent=indent)
