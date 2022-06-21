# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod

import h5py
import numpy as np

from osb.extensions.data_set import Context, HDF5DataSet, BigANNVectorDataSet

""" Module containing utility classes and functions for working with data sets.

Included are utilities that can be used to build data sets and write them to 
paths.
"""


class DataSetBuildContext:
    """ Data class capturing information needed to build a particular data set

    Attributes:
        data_set_context: Indicator of what the data set is used for,
        vectors: A 2D array containing vectors that are used to build data set.
        path: string representing path where data set should be serialized to.
    """
    def __init__(self, data_set_context: Context, vectors: np.ndarray, path: str):
        self.data_set_context: Context = data_set_context
        self.vectors: np.ndarray = vectors  #TODO: Validate shape
        self.path: str = path

    def get_num_vectors(self) -> int:
        return self.vectors.shape[0]

    def get_dimension(self) -> int:
        return self.vectors.shape[1]

    def get_type(self) -> np.dtype:
        return self.vectors.dtype


class DataSetBuilder(ABC):
    """ Abstract builder used to create a build a collection of data sets

    Attributes:
        data_set_build_contexts: list of data set build contexts that builder
                                 will build.
    """
    def __init__(self):
        self.data_set_build_contexts = list()

    def add_data_set_build_context(self, data_set_build_context: DataSetBuildContext):
        """ Adds a data set build context to list of contexts to be built.

        Args:
            data_set_build_context: DataSetBuildContext to be added to list

        Returns: Updated DataSetBuilder

        """
        self._validate_data_set_context(data_set_build_context)
        self.data_set_build_contexts.append(data_set_build_context)
        return self

    def build(self):
        """ Builds and serializes all data sets build contexts

        Returns:

        """
        [self._build_data_set(data_set_build_context) for data_set_build_context
         in self.data_set_build_contexts]

    @abstractmethod
    def _build_data_set(self, context: DataSetBuildContext):
        """ Builds an individual data set

        Args:
            context: DataSetBuildContext of data set to be built

        Returns:

        """
        pass

    @abstractmethod
    def _validate_data_set_context(self, context: DataSetBuildContext):
        """ Validates that data set context can be added to this builder

        Args:
            context: DataSetBuildContext to be validated

        Returns:

        """
        pass


class HDF5Builder(DataSetBuilder):

    def __init__(self):
        super(HDF5Builder, self).__init__()
        self.data_set_meta_data = dict()

    def _validate_data_set_context(self, context: DataSetBuildContext):
        if context.path not in self.data_set_meta_data.keys():
            self.data_set_meta_data[context.path] = {
                context.data_set_context: context
            }
            return

        if context.data_set_context in \
                self.data_set_meta_data[context.path].keys():
            raise IllegalDataSetBuildContext("Path and context for data set "
                                             "are already present in builder.")

        self.data_set_meta_data[context.path][context.data_set_context] = \
            context

    @staticmethod
    def _validate_extension(context: DataSetBuildContext):
        ext = context.path.split('.')[-1]

        if ext != HDF5DataSet.FORMAT_NAME:
            raise IllegalDataSetBuildContext("Invalid file extension")

    def _build_data_set(self, context: DataSetBuildContext):
        # For HDF5, because multiple data sets can be grouped in the same file,
        # we will build data sets in memory and not write to disk until
        # _flush_data_sets_to_disk is called
        with h5py.File(context.path, 'a') as hf:
            hf.create_dataset(
                HDF5DataSet.parse_context(context.data_set_context),
                data=context.vectors
            )


class BigANNBuilder(DataSetBuilder):

    def _validate_data_set_context(self, context: DataSetBuildContext):
        self._validate_extension(context)

        # prevent the duplication of paths for data sets
        data_set_paths = [c.path for c in self.data_set_build_contexts]
        if any(data_set_paths.count(x) > 1 for x in data_set_paths):
            raise IllegalDataSetBuildContext("Build context paths have to be "
                                              "unique.")

    @staticmethod
    def _validate_extension(context: DataSetBuildContext):
        ext = context.path.split('.')[-1]

        if ext != BigANNVectorDataSet.U8BIN_EXTENSION and ext != \
                BigANNVectorDataSet.FBIN_EXTENSION:
            raise IllegalDataSetBuildContext("Invalid file extension")

        if ext == BigANNVectorDataSet.U8BIN_EXTENSION and context.get_type() != \
                np.u8int:
            raise IllegalDataSetBuildContext("Invalid data type for {} ext."
                                             .format(BigANNVectorDataSet
                                                     .U8BIN_EXTENSION))

        if ext == BigANNVectorDataSet.FBIN_EXTENSION and context.get_type() != \
                np.float32:
            print(context.get_type())
            raise IllegalDataSetBuildContext("Invalid data type for {} ext."
                                             .format(BigANNVectorDataSet
                                                     .FBIN_EXTENSION))

    def _build_data_set(self, context: DataSetBuildContext):
        num_vectors = context.get_num_vectors()
        dimension = context.get_dimension()

        with open(context.path, 'wb') as f:
            f.write(int.to_bytes(num_vectors, 4, "little"))
            f.write(int.to_bytes(dimension, 4, "little"))
            context.vectors.tofile(f)


def create_random_2d_array(num_vectors: int, dimension: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.random(size=(num_vectors, dimension), dtype=np.float32)


class IllegalDataSetBuildContext(Exception):
    """Exception raised when passed in DataSetBuildContext is illegal

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str):
        self.message = f'{message}'
        super().__init__(self.message)

