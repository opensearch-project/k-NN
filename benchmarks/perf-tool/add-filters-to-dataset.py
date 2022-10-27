import getopt
import os
import random
import sys

import h5py

from osb.extensions.data_set import Context, HDF5DataSet

"""
Script builds complex dataset with additional attributes from exiting dataset that has only vectors. 
Additional attributes are predefined in the script: color, taste, age. Only HDF5 format of vector dataset is supported.

Script generates additional dataset of neighbours (ground truth) for each filter type. 

Example of usage:
    
    create new hdf5 file with attribute dataset
    add-filters-to-dataset.py ~/dev/opensearch/k-NN/benchmarks/perf-tool/dataset/data.hdf5 ~/dev/opensearch/datasets/data-with-attr True False
    
    create new hdf5 file with filter datasets
    add-filters-to-dataset.py ~/dev/opensearch/k-NN/benchmarks/perf-tool/dataset/data-with-attr.hdf5 ~/dev/opensearch/datasets/data-with-filters False True
"""

class Dataset():
    DEFAULT_INDEX_NAME = "test-index"
    DEFAULT_FIELD_NAME = "test-field"
    DEFAULT_CONTEXT = Context.INDEX
    DEFAULT_TYPE = HDF5DataSet.FORMAT_NAME
    DEFAULT_NUM_VECTORS = 10
    DEFAULT_DIMENSION = 10
    DEFAULT_RANDOM_STRING_LENGTH = 8

    def createDataset(self, source_dataset_path, out_file_path, generate_attrs: bool, generate_filters: bool) -> None:
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
                        and  (attributes[vector_idx][1].decode() == 'bitter') \
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
        perc = float(filtered_count/overall_count) * 100
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

    worker = Dataset()
    worker.createDataset(in_file_path, out_file_path, generate_attr, generate_filters)

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
   main(sys.argv[1:])

