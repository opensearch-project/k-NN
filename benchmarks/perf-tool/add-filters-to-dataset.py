# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
"""
Script builds complex dataset with additional attributes from exiting dataset that has only vectors. 
Additional attributes are predefined in the script: color, taste, age. Only HDF5 format of vector dataset is supported.

Output dataset file will have additional dataset 'attributes' with multiple columns, each column corresponds to one attribute
from an attribute set, and value is generated at random, e.g.:

0: green	None	71
1: green	bitter	28

there is no explicit index reference in 'attributes' dataset, index of the row corresponds to a document id. 
For instance, in example above two rows of fields mapped to documents with ids '0' and '1'.  

If 'generate_filters' flag is set script generates additional dataset of neighbours (ground truth) for each filter type. 
Output is a new file with several datasets, each dataset corresponds to one filter. Datasets are named 'neighbour_filter_X'
where X is 1 based index of particular filter. 
Each dataset has rows with array of integers, where integer corresponds to 
a document id from original dataset with additional fields. Array ca have -1 values that are treated as null, this is because
subset of filtered documents is same of smaller than original set. 

For example, dataset file content may look like :

neighbour_filter_1: [[ 2,  5, -1],
                     [ 3,  1, -1],
                     [ 2   5,  7]]
neighbour_filter_2: [[-1, -1, -1],
                     [ 5,  6, -1],
                     [ 4,  2,  1]]

In this case we do have datasets for two filters, 3 query results for each. [2, 5, -1] indicates that for first query 
if filter 1 is used most similar document is with id 2, next similar is 5, and the rest do not pass filter 1 criteria.

Example of script usage:
    
    create new hdf5 file with attribute dataset
    add-filters-to-dataset.py ~/dev/opensearch/k-NN/benchmarks/perf-tool/dataset/data.hdf5 ~/dev/opensearch/datasets/data-with-attr True False
    
    create new hdf5 file with filter datasets
    add-filters-to-dataset.py ~/dev/opensearch/k-NN/benchmarks/perf-tool/dataset/data-with-attr.hdf5 ~/dev/opensearch/datasets/data-with-filters False True
"""

import getopt
import os
import random
import sys

import h5py

from osb.extensions.data_set import HDF5DataSet


class _Dataset:
    """Type of dataset container for data with additional attributes"""
    DEFAULT_TYPE = HDF5DataSet.FORMAT_NAME

    def create_dataset(self, source_dataset_path, out_file_path, generate_attrs: bool, generate_filters: bool) -> None:
        path_elements = os.path.split(os.path.abspath(source_dataset_path))
        data_set_dir = path_elements[0]

        # For HDF5, because multiple data sets can be grouped in the same file,
        # we will build data sets in memory and not write to disk until
        # _flush_data_sets_to_disk is called
        # read existing dataset
        data_hdf5 = os.path.join(os.path.dirname(os.path.realpath('/')), source_dataset_path)

        with h5py.File(data_hdf5, "r") as hf:

            if generate_attrs:
                data_set_w_attr = self.create_dataset_file(out_file_path, self.DEFAULT_TYPE, data_set_dir)

                possible_colors = ['red', 'green', 'yellow', 'blue', None]
                possible_tastes = ['sweet', 'salty', 'sour', 'bitter', None]
                max_age = 100

                for key in hf.keys():
                    if key not in ['neighbors', 'test', 'train']:
                        continue
                    data_set_w_attr.create_dataset(key, data=hf[key][()])

                attributes = []
                for i in range(len(hf['train'])):
                    attr = [random.choice(possible_colors), random.choice(possible_tastes),
                            random.randint(0, max_age + 1)]
                    attributes.append(attr)

                data_set_w_attr.create_dataset('attributes', (len(attributes), 3), 'S10', data=attributes)

                data_set_w_attr.flush()
                data_set_w_attr.close()

            if generate_filters:
                attributes = hf['attributes'][()]
                expected_neighbors = hf['neighbors'][()]

                data_set_filters = self.create_dataset_file(out_file_path, self.DEFAULT_TYPE, data_set_dir)

                def filter1(attributes, vector_idx):
                    if attributes[vector_idx][0].decode() == 'red' and int(attributes[vector_idx][2].decode()) >= 20:
                        return True
                    else:
                        return False

                self.apply_filter(expected_neighbors, attributes, data_set_filters, 'neighbors_filter_1', filter1)

                # filter 2 - color = blue or None and taste = 'salty'
                def filter2(attributes, vector_idx):
                    if (attributes[vector_idx][0].decode() == 'blue' or attributes[vector_idx][
                        0].decode() == 'None') and attributes[vector_idx][1].decode() == 'salty':
                        return True
                    else:
                        return False

                self.apply_filter(expected_neighbors, attributes, data_set_filters, 'neighbors_filter_2', filter2)

                # filter 3 - color and taste are not None and age is between 20 and 80
                def filter3(attributes, vector_idx):
                    if attributes[vector_idx][0].decode() != 'None' and attributes[vector_idx][
                        1].decode() != 'None' and 20 <= \
                            int(attributes[vector_idx][2].decode()) <= 80:
                        return True
                    else:
                        return False

                self.apply_filter(expected_neighbors, attributes, data_set_filters, 'neighbors_filter_3', filter3)

                # filter 4 - color green or blue and taste is bitter and age is between (30, 60)
                def filter4(attributes, vector_idx):
                    if (attributes[vector_idx][0].decode() == 'green' or attributes[vector_idx][0].decode() == 'blue') \
                            and (attributes[vector_idx][1].decode() == 'bitter') \
                            and 30 <= int(attributes[vector_idx][2].decode()) <= 60:
                        return True
                    else:
                        return False

                self.apply_filter(expected_neighbors, attributes, data_set_filters, 'neighbors_filter_4', filter4)

                # filter 5 color is (green or blue or yellow) or taste = sweet or age is between (30, 70)
                def filter5(attributes, vector_idx):
                    if attributes[vector_idx][0].decode() == 'green' or attributes[vector_idx][0].decode() == 'blue' \
                            or attributes[vector_idx][0].decode() == 'yellow' \
                            or attributes[vector_idx][1].decode() == 'sweet' \
                            or 30 <= int(attributes[vector_idx][2].decode()) <= 70:
                        return True
                    else:
                        return False

                self.apply_filter(expected_neighbors, attributes, data_set_filters, 'neighbors_filter_5', filter5)

                data_set_filters.flush()
                data_set_filters.close()

    def apply_filter(self, expected_neighbors, attributes, data_set_w_filtering, filter_name, filter_func):
        neighbors_filter = []
        filtered_count = 0
        for expected_neighbors_row in expected_neighbors:
            neighbors_filter_row = [-1] * len(expected_neighbors_row)
            idx = 0
            for vector_idx in expected_neighbors_row:
                if filter_func(attributes, vector_idx):
                    neighbors_filter_row[idx] = vector_idx
                    idx += 1
                    filtered_count += 1
            neighbors_filter.append(neighbors_filter_row)
        overall_count = len(expected_neighbors) * len(expected_neighbors[0])
        perc = float(filtered_count / overall_count) * 100
        print('ground truth size for {} is {}, percentage {}'.format(filter_name, filtered_count, perc))
        data_set_w_filtering.create_dataset(filter_name, data=neighbors_filter)
        return expected_neighbors

    def create_dataset_file(self, file_name, extension, data_set_dir) -> h5py.File:
        data_set_file_name = "{}.{}".format(file_name, extension)
        data_set_path = os.path.join(data_set_dir, data_set_file_name)

        data_set_w_filtering = h5py.File(data_set_path, 'a')

        return data_set_w_filtering


def main(argv):
    opts, args = getopt.getopt(argv, "")
    in_file_path = args[0]
    out_file_path = args[1]
    generate_attr = str2bool(args[2])
    generate_filters = str2bool(args[3])

    worker = _Dataset()
    worker.create_dataset(in_file_path, out_file_path, generate_attr, generate_filters)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    main(sys.argv[1:])
