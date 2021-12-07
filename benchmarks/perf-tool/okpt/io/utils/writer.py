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
    return open(path, 'w')


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
