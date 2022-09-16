import os
import random


import h5py
import numpy as np

from osb.extensions.data_set import Context, HDF5DataSet
from osb.tests.data_set_helper import DataSetBuildContext

class HDF5Dataset():
    DEFAULT_INDEX_NAME = "test-index"
    DEFAULT_FIELD_NAME = "test-field"
    DEFAULT_CONTEXT = Context.INDEX
    DEFAULT_TYPE = HDF5DataSet.FORMAT_NAME
    DEFAULT_NUM_VECTORS = 10
    DEFAULT_DIMENSION = 10
    DEFAULT_RANDOM_STRING_LENGTH = 8

    def create_random_2d_array(self, num_vectors: int, dimension: int) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.random(size=(num_vectors, dimension), dtype=np.float32)

    def createDataset(self) -> None:
            self.data_set_dir = os.getcwd()

            # Create a data set we know to be valid for convenience
            self.valid_data_set_path = self._create_data_set(
                self.DEFAULT_NUM_VECTORS,
                self.DEFAULT_DIMENSION,
                self.DEFAULT_TYPE,
                self.DEFAULT_CONTEXT,
                self.data_set_dir
            )

    def _create_data_set(
            self,
            num_vectors: int,
            dimension: int,
            extension: str,
            data_set_context: Context,
            data_set_dir
    ) -> str:

        #file_name_base = ''.join(random.choice(string.ascii_letters) for _ in
        #                         range(self.DEFAULT_RANDOM_STRING_LENGTH))
        file_name_base = 'data-with-attr'
        data_set_file_name = "{}.{}".format(file_name_base, extension)
        data_set_path = os.path.join(data_set_dir, data_set_file_name)
        context = DataSetBuildContext(
            data_set_context,
            self.create_random_2d_array(num_vectors, dimension),
            data_set_path)


        self._build_data_set(context)

        return data_set_path

    def _build_data_set(self, context: DataSetBuildContext):
        # For HDF5, because multiple data sets can be grouped in the same file,
        # we will build data sets in memory and not write to disk until
        # _flush_data_sets_to_disk is called
        # read existing dataset

        data_set_w_filtering = h5py.File(context.path, 'a')

        data_hdf5 = os.path.join(os.path.dirname(os.path.realpath('/')),
                                 'Users/gaievski/dev/opensearch/datasets/data.hdf5')
        #data_hdf5 = os.path.join(os.path.dirname(os.path.realpath('/')),
        #                         'Users/gaievski/dev/opensearch/datasets/glove-25-angular.hdf5')
        with h5py.File(data_hdf5, "r") as hf:
            print("Keys: %s" % hf.keys())

            for key in hf.keys():
                print(key)
                if key not in ['neighbors', 'test', 'train']:
                    continue
                data_set_w_filtering.create_dataset(key, data = hf[key][()])

            possible_colors = ['red', 'green', 'yellow', 'blue', None]
            possible_tastes = ['sweet', 'salty', 'sour', 'bitter', None]
            max_age = 100
            attributes = []
            for i in range(len(hf['train'])):
                attr = [random.choice(possible_colors), random.choice(possible_tastes), random.randint(0, max_age + 1)]
                attributes.append(attr)

            data_set_w_filtering.create_dataset('attributes', (len(attributes), 3), 'S10', data=attributes)
            expected_neighbors = hf['neighbors'][()]

            def filter1(attributes, vector_idx):
                if attributes[vector_idx][0] == 'red' and attributes[vector_idx][2] >= 20:
                    return True
                else:
                    return False

            self.apply_filter(expected_neighbors, attributes, data_set_w_filtering, 'neighbors_filter_1', filter1)

            # filter 2 - color = blue or None and taste = 'salty'
            def filter2(attributes, vector_idx):
                if (attributes[vector_idx][0] == 'blue' or attributes[vector_idx][0] == 'None') and attributes[vector_idx][1] == 'salty':
                    return True
                else:
                    return False

            self.apply_filter(expected_neighbors, attributes, data_set_w_filtering, 'neighbors_filter_2', filter2)

            # filter 3 - color and taste are not None and age is between 20 and 80
            def filter3(attributes, vector_idx):
                if attributes[vector_idx][0] != 'None' and attributes[vector_idx][1] != 'None' and 20 <= \
                        attributes[vector_idx][2] <= 80:
                    return True
                else:
                    return False

            self.apply_filter(expected_neighbors, attributes, data_set_w_filtering, 'neighbors_filter_3', filter3)

            data_set_w_filtering.flush()
            data_set_w_filtering.close()

        """
        with h5py.File(context.path, 'a') as hf:
            hf.create_dataset(
                HDF5DataSet.parse_context(context.data_set_context),
                data=context.vectors
            )
        """
        """
        with h5py.File(context.path, 'a') as hf:
            colors = ['red',None, 'green', None, 'red']
            #asciiList = [self.dd(n) for n in strList]
            hf.create_dataset('color', (len(colors), 1), 'S10', colors)
            taste = ['sweet', 'sweet', 'sour', 'salty', None]
            # asciiList = [self.dd(n) for n in strList]
            hf.create_dataset('taste', (len(taste), 1), 'S10', taste)
            hf.create_dataset(
                HDF5DataSet.parse_context(context.data_set_context),
                data=context.vectors
            )
            hf.flush()
            hf.close()

        with h5py.File(context.path, "r") as hf:
            print("Keys: %s" % hf.keys())

            for key in hf.keys():
                print(key)
                data = hf[key][()]
                for d in data:
                    #print(type(d))
                    if key in ['color', 'taste']:
                        s = d[0].decode()
                        if s == "None":
                            print("Empty string")
                        else:
                            print(s)
                    else:
                        print(d)
        """

    def apply_filter(self, expected_neighbors, attributes, data_set_w_filtering, filter_name, filter_func):
        # filter one - color = red, age >= 20
        neighbors_filter_1 = []
        for expected_neighbors_row in expected_neighbors:
            neighbors_filter_1_row = [-1] * len(expected_neighbors_row)
            idx = 0
            for vector_idx in expected_neighbors_row:
                #if attributes[vector_idx][0] == 'red' and attributes[vector_idx][2] >= 20:
                if filter_func(attributes, vector_idx):
                    neighbors_filter_1_row[idx] = vector_idx
                    idx += 1
            neighbors_filter_1.append(neighbors_filter_1_row)
        data_set_w_filtering.create_dataset(filter_name, data=neighbors_filter_1)
        return expected_neighbors

    def dd(self, n):
        return None if n is None else n.encode("ascii", "ignore")

worker = HDF5Dataset()
worker.createDataset()