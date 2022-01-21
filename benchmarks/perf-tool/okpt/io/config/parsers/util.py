# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Utility functions for parsing"""


from okpt.io.config.parsers.base import ConfigurationError
from okpt.io.dataset import HDF5DataSet, BigANNNeighborDataSet, \
    BigANNVectorDataSet, DataSet, Context


def parse_dataset(dataset_format: str, dataset_path: str,
                  context: Context) -> DataSet:
    if dataset_format == 'hdf5':
        return HDF5DataSet(dataset_path, context)

    if dataset_format == 'bigann' and context == Context.NEIGHBORS:
        return BigANNNeighborDataSet(dataset_path)

    if dataset_format == 'bigann':
        return BigANNVectorDataSet(dataset_path)

    raise Exception("Unsupported data-set format")


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
