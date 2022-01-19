# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Utility functions for parsing"""

from dataclasses import dataclass
from typing import Union, cast
import h5py

from okpt.io.config.parsers.base import ConfigurationError


class DataSet(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self, chunk_size: int):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def size(self):
        pass


class HDF5DataSet(Dataset):

    def __init__(self, dataset_path: str, selector: str):
        file = h5py.File(dataset_path)
        self.data = cast(h5py.Dataset, file[selector])
        self.current = 0

    def read(self, chunk_size: int):
        ##TODO: This should only return the maximum number of results
        v = cast(np.ndarray, self.data[self.current:self.current + chunk_size])
        self.current += chunk_size
        return v

    def get_data(self):
        return self.data

    def size(self):
        return self.data.len()


def parse_dataset(dataset_format: str, dataset_path: str,
                  selector: str = None) -> DataSet:
    if dataset_format == 'hdf5':
        return HDF5DataSet(dataset_path, selector)

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
