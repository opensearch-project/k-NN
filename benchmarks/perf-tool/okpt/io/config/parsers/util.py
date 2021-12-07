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
"""Utility functions for parsing"""

from dataclasses import dataclass
from typing import Union, cast
import h5py

from okpt.io.config.parsers.base import ConfigurationError


@dataclass
class Dataset:
    train: h5py.Dataset
    test: h5py.Dataset
    neighbors: h5py.Dataset
    distances: h5py.Dataset


def parse_dataset(dataset_path: str, dataset_format: str) -> Union[Dataset]:
    if dataset_format == 'hdf5':
        file = h5py.File(dataset_path)
        return Dataset(train=cast(h5py.Dataset, file['train']),
                       test=cast(h5py.Dataset, file['test']),
                       neighbors=cast(h5py.Dataset, file['neighbors']),
                       distances=cast(h5py.Dataset, file['distances']))
    else:
        raise Exception()


def parse_string_param(key: str, first_map, second_map, default) -> str:
    value = first_map.get(key)
    if value is not None:
        if type(value) is str:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    value = second_map.get(key)
    if value is not None:
        if type(value) is str:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    if default is None:
        raise ConfigurationError("{} must be set".format(key))
    return default


def parse_int_param(key: str, first_map, second_map, default) -> int:
    value = first_map.get(key)
    if value is not None:
        if type(value) is int:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    value = second_map.get(key)
    if value is not None:
        if type(value) is int:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    if default is None:
        raise ConfigurationError("{} must be set".format(key))
    return default


def parse_bool_param(key: str, first_map, second_map, default) -> bool:
    value = first_map.get(key)
    if value is not None:
        if type(value) is bool:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    value = second_map.get(key)
    if value is not None:
        if type(value) is bool:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    if default is None:
        raise ConfigurationError("{} must be set".format(key))
    return default


def parse_dict_param(key: str, first_map, second_map, default) -> dict:
    value = first_map.get(key)
    if value is not None:
        if type(value) is dict:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    value = second_map.get(key)
    if value is not None:
        if type(value) is dict:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    if default is None:
        raise ConfigurationError("{} must be set".format(key))
    return default


def parse_list_param(key: str, first_map, second_map, default) -> list:
    value = first_map.get(key)
    if value is not None:
        if type(value) is list:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    value = second_map.get(key)
    if value is not None:
        if type(value) is list:
            return value
        raise ConfigurationError("Invalid type for {}".format(key))

    if default is None:
        raise ConfigurationError("{} must be set".format(key))
    return default
