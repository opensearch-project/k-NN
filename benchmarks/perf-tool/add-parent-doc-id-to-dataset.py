#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

"""
Script builds complex dataset with additional attributes from exiting dataset that has only vectors.
Additional attributes are predefined in the script: color, taste, age, and parent doc id. Only HDF5 format of vector dataset is supported.

Output dataset file will have additional dataset 'attributes' with multiple columns, each column corresponds to one attribute
from an attribute set, and value is generated at random, e.g.:

0: green	None	71  1
1: green	bitter	28  1
2: green	bitter	28  1
3: green	bitter	28  2
...

there is no explicit index reference in 'attributes' dataset, index of the row corresponds to a document id.
For instance, in example above two rows of fields mapped to documents with ids '0' and '1'.

The parend doc ids are assigned in non-decreasing order.

If 'generate_filters' flag is set script generates additional dataset of neighbours (ground truth).
Output is a new file with three dataset each of which corresponds to a certain type of query.
Dataset name neighbour_nested is a ground truth for query without filtering.
Dataset name neighbour_filtered_relaxed is a ground truth for query with filtering of (30 <= age <= 70) or color in ["green", "blue", "yellow"] or taste in ["sweet"]
Dataset name neighbour_filtered_restrictive is a ground truth for query with filtering of (30 <= age <= 60) and color in ["green", "blue"] and taste in ["bitter"]


Each dataset has rows with array of integers, where integer corresponds to
a document id from original dataset with additional fields.

Example of script usage:

    create new hdf5 file with attribute dataset
    add-parent-doc-id-to-dataset.py ~/dev/opensearch/k-NN/benchmarks/perf-tool/dataset/data.hdf5 ~/dev/opensearch/datasets/data-nested.hdf5

"""
import getopt
import multiprocessing
import random
import sys
from multiprocessing import Process
from typing import cast
import traceback

import h5py
import numpy as np


class MyVector:
    def __init__(self, vector, id, color=None, taste=None, age=None, parent_id=None):
        self.vector = vector
        self.id = id
        self.age = age
        self.color = color
        self.taste = taste
        self.parent_id = parent_id

    def apply_restricted_filter(self):
        return (30 <= self.age <= 60) and self.color in ["green", "blue"] and self.taste in ["bitter"]

    def apply_relaxed_filter(self):
        return (30 <= self.age <= 70) or self.color in ["green", "blue", "yellow"] or self.taste in ["sweet"]

    def __str__(self):
        return f'Vector : {self.vector}, id : {self.id}, color: {self.color}, taste: {self.taste}, age: {self.age}, parent_id: {self.parent_id}\n'

    def __repr__(self):
        return f'Vector : {self.vector}, id : {self.id}, color: {self.color}, taste: {self.taste}, age: {self.age}, parent_id: {self.parent_id}\n'

class HDF5DataSet:
    def __init__(self, file_path, key):
        self.file_name = file_path
        self.file = h5py.File(self.file_name)
        self.key = key
        self.data = cast(h5py.Dataset, self.file[key])
        self.metadata = None
        self.metadata = cast(h5py.Dataset, self.file["attributes"]) if key == "train" else None
        print(f'Keys in the file are {self.file.keys()}')

    def read(self, start, end=None):
        if end is None:
            end = self.data.len()
        values = cast(np.ndarray, self.data[start:end])
        metadata = cast(list, self.metadata[start:end]) if self.metadata is not None else None
        if metadata is not None:
            print(metadata)
        vectors = []
        i = 0
        for value in values:
            if self.metadata is None:
                vector = MyVector(value, i)
            else:
                # color, taste, age, and parent id
                vector = MyVector(value, i, str(metadata[i][0].decode()), str(metadata[i][1].decode()),
                                  int(metadata[i][2]), int(metadata[i][3]))
            vectors.append(vector)
            i = i + 1
        return vectors

    def read_neighbors(self, start, end):
        return cast(np.ndarray, self.data[start:end])

    def size(self):
        return self.data.len()

    def close(self):
        self.file.close()

class _Dataset:
    def run(self, source_path, target_path) -> None:
        # Add attributes
        print(f'Adding attributes started.')
        with h5py.File(source_path, "r") as in_file:
            out_file = h5py.File(target_path, "w")
            possible_colors = ['red', 'green', 'yellow', 'blue', None]
            possible_tastes = ['sweet', 'salty', 'sour', 'bitter', None]
            max_age = 100
            min_field_size = 1000
            max_field_size = 10001

            # Copy train and test data
            for key in in_file.keys():
                if key not in ['test', 'train']:
                    continue
                out_file.create_dataset(key, data=in_file[key][()])

            # Generate attributes
            attributes = []
            field_size = random.randint(min_field_size, max_field_size)
            parent_id = 1
            field_count = 0
            for i in range(len(in_file['train'])):
                attr = [random.choice(possible_colors), random.choice(possible_tastes),
                        random.randint(0, max_age + 1), parent_id]
                attributes.append(attr)
                field_count += 1
                if field_count >= field_size:
                    field_size = random.randint(min_field_size, max_field_size)
                    field_count = 0
                    parent_id += 1
            out_file.create_dataset('attributes', (len(attributes), 4), 'S10', data=attributes)

            out_file.flush()
            out_file.close()

        print(f'Adding attributes completed.')


        # Calculate ground truth
        print(f'Calculating ground truth started.')
        cpus = multiprocessing.cpu_count()
        total_clients = min(8, cpus)  # 1  # 10
        hdf5Data_train = HDF5DataSet(target_path, "train")
        train_vectors = hdf5Data_train.read(0, hdf5Data_train.size())
        hdf5Data_train.close()
        print(f'Train vector size: {len(train_vectors)}')

        hdf5Data_test = HDF5DataSet(target_path, "test")
        total_queries = hdf5Data_test.size()  # 10000
        dis = [] * total_queries

        for i in range(total_queries):
            dis.insert(i, [])

        queries_per_client = int(total_queries / total_clients + 0.5)
        if queries_per_client == 0:
            queries_per_client = total_queries

        processes = []
        test_vectors = hdf5Data_test.read(0, total_queries)
        hdf5Data_test.close()
        tasks_that_are_done = multiprocessing.Queue()
        for client in range(total_clients):
            start_index = int(client * queries_per_client)
            if start_index + queries_per_client <= total_queries:
                end_index = int(start_index + queries_per_client)
            else:
                end_index = total_queries

            print(f'Start Index: {start_index}, end Index: {end_index}')
            print(f'client is  : {client}')
            p = Process(target=queryTask, args=(
                train_vectors, test_vectors, start_index, end_index, client, total_queries, tasks_that_are_done))
            processes.append(p)
            p.start()
            if end_index >= total_queries:
                print(f'Exiting end Index : {end_index} total_queries: {total_queries}')
                break

        # wait for tasks to be completed
        print('Waiting for all tasks to be completed')
        j = 0
        # This is required because threads can hang if the data sent from the sub process increases by a certain limit
        # https://stackoverflow.com/questions/21641887/python-multiprocessing-process-hangs-on-join-for-large-queue
        while j < total_queries:
            while not tasks_that_are_done.empty():
                calculatedDis = tasks_that_are_done.get()
                i = 0
                for d in calculatedDis:
                    if d:
                        dis[i] = d
                        j = j + 1
                    i = i + 1

        for p in processes:
            if p.is_alive():
                p.join()
            else:
                print("Process was not alive hence shutting down")

        data_set_file = h5py.File(target_path, "a")
        for type in ['nested', 'relaxed', 'restricted']:
            results = []
            for d in dis:
                r = []
                for i in range(min(10000, len(d[type]))):
                    r.append(d[type][i]['id'])
                results.append(r)


            data_set_file.create_dataset("neighbour_" + type, (len(results), len(results[0])), data=results)
        data_set_file.flush()
        data_set_file.close()

def calculateL2Distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def queryTask(train_vectors, test_vectors, startIndex, endIndex, process_number, total_queries, tasks_that_are_done):
    print(f'Starting Process number : {process_number}')
    all_distances = [] * total_queries
    for i in range(total_queries):
        all_distances.insert(i, {})
    try:
        test_vectors = test_vectors[startIndex:endIndex]
        i = startIndex
        for test in test_vectors:
            distances = []
            values = {}
            for value in train_vectors:
                values[value.id] = value
                distances.append({
                    "dis": calculateL2Distance(test.vector, value.vector),
                    "id": value.parent_id
                })

            distances.sort(key=lambda vector: vector['dis'])
            seen_set_nested = set()
            seen_set_restricted = set()
            seen_set_relaxed = set()
            nested = []
            restricted = []
            relaxed = []
            for sub_i in range(len(distances)):
                id = distances[sub_i]['id']
                # Check if the number has been seen before
                if len(nested) < 1000 and id not in seen_set_nested:
                    # If not seen before, mark it as seen
                    seen_set_nested.add(id)
                    nested.append(distances[sub_i])
                if len(restricted) < 1000 and id not in seen_set_restricted and values[id].apply_restricted_filter():
                    seen_set_restricted.add(id)
                    restricted.append(distances[sub_i])
                if len(relaxed) < 1000 and id not in seen_set_relaxed and values[id].apply_relaxed_filter():
                    seen_set_relaxed.add(id)
                    relaxed.append(distances[sub_i])

            all_distances[i]['nested'] = nested
            all_distances[i]['restricted'] = restricted
            all_distances[i]['relaxed'] = relaxed
            print(f"Process {process_number} queries completed: {i + 1 - startIndex}, queries left: {endIndex - i - 1}")
            i = i + 1
    except:
        print(
            f"Got exception while running the thread: {process_number} with startIndex: {startIndex} endIndex: {endIndex} ")
        traceback.print_exc()
    tasks_that_are_done.put(all_distances)
    print(f'Exiting Process number : {process_number}')


def main(argv):
    opts, args = getopt.getopt(argv, "")
    in_file_path = args[0]
    out_file_path = args[1]

    worker = _Dataset()
    worker.run(in_file_path, out_file_path)

if __name__ == "__main__":
    main(sys.argv[1:])