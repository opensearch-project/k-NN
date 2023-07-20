#  SPDX-License-Identifier: Apache-2.0
#
#  The OpenSearch Contributors require contributions made to
#  this file be licensed under the Apache-2.0 license or a
#  compatible open source license.
#
#  Modifications Copyright OpenSearch Contributors. See
#  GitHub history for details.
import os
import getopt
import sys
from multiprocessing import Process
from typing import cast
import numpy as np
import traceback
import h5py
import multiprocessing

cpus = multiprocessing.cpu_count()


class MyVector:

    def __init__(self, vector, id, color=None, taste=None, age=None):
        self.vector = vector
        self.id = id
        self.age = age
        self.color = color
        self.taste = taste

    def apply_filter(self):
        return (30 <= self.age <= 70) or self.color in ["green", "blue", "yellow"] or self.taste in ["sweet"]

    def __str__(self):
        return f'Vector : {self.vector}, id : {self.id}, color: {self.color}, taste: {self.taste}, age: {self.age}\n'

    def __repr__(self):
        return f'Vector : {self.vector}, id : {self.id}, color: {self.color}, taste: {self.taste}, age: {self.age}\n'


class HDF5DataSet:

    def __init__(self, file_path, key):
        self.file_name = file_path
        self.file = h5py.File(self.file_name)
        self.key = key
        self.data = cast(h5py.Dataset, self.file[key])
        self.metadata = cast(h5py.Dataset, self.file["attributes"]) if key == "train" else None
        print(f'Keys in the file are {self.file.keys()}')

    def read(self, start, end):
        values = cast(np.ndarray, self.data[start:end])
        metadata = cast(list, self.metadata[start:end]) if self.metadata is not None else None
        print(f'Given length {end - start}, actual length : {len(values)}')
        if metadata is not None:
            print(metadata)
        vectors = []
        i = 0
        for value in values:
            if self.metadata is None:
                vector = MyVector(value, i)
            else:
                # color, taste, age
                vector = MyVector(value, i, str(metadata[i][0].decode()), str(metadata[i][1].decode()),
                                  int(metadata[i][2]))
            vectors.append(vector)
            i = i + 1
        return vectors

    def read_neighbors(self, start, end):
        return cast(np.ndarray, self.data[start:end])


def calculateL2Distance(point1, point2):
    return np.linalg.norm(point1 - point2)


filter5 = {
    "bool":
        {
            "should":
                [
                    {
                        "range":
                            {
                                "age":
                                    {
                                        "gte": 30,
                                        "lte": 70
                                    }
                            }
                    },
                    {
                        "term":
                            {
                                "color": "green"
                            }
                    },
                    {
                        "term":
                            {
                                "color": "blue"
                            }
                    },
                    {
                        "term":
                            {
                                "color": "yellow"
                            }
                    },
                    {
                        "term":
                            {
                                "color": "sweet"
                            }
                    }
                ]
        }
}


def create_dataset_file(file_name, data_set_dir) -> h5py.File:
    data_set_file_name = "{}.{}".format(file_name, "hdf5")
    data_set_path = os.path.join(data_set_dir, data_set_file_name)

    data_set_w_filtering = h5py.File(data_set_path, 'a')

    return data_set_w_filtering


def main(argv):
    opts, args = getopt.getopt(argv, "")
    in_file_path = args[0]
    out_file_path = args[1]
    if out_file_path is None:
        print("out_file_path is not provided")
        return
    total_clients = min(8, cpus)  # 1  # 10
    hdf5Data_train = HDF5DataSet(os.path.join(in_file_path, "sift-128-euclidean-with-attr.hdf5"), "train")
    train_vectors = hdf5Data_train.read(0, 1000000)
    print(f'Train vector size: {len(train_vectors)}')

    hdf5Data_test = HDF5DataSet(os.path.join(in_file_path, "sift-128-euclidean-with-attr.hdf5"), "test")
    total_queries = 8  # 10000
    dis = [] * total_queries

    for i in range(total_queries):
        dis.insert(i, [])

    queries_per_client = int(total_queries / total_clients)
    if queries_per_client == 0:
        queries_per_client = total_queries

    processes = []
    test_vectors = hdf5Data_test.read(0, total_queries)
    tasks_that_are_done = multiprocessing.Queue()
    for client in range(total_clients):
        start_index = int(client * queries_per_client)
        if start_index + queries_per_client <= total_queries:
            end_index = int(start_index + queries_per_client)
        else:
            end_index = total_queries - start_index

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
                    print("Dis is not null")
                    dis[i] = d
                    j = j + 1
                i = i + 1

    for p in processes:
        if p.is_alive():
            p.join()
        else:
            print("Process was not alive hence shutting down")


    validationDataSet = HDF5DataSet(os.path.join(in_file_path, "sift-128-euclidean-with-filters.hdf5"),
                                    "neighbors_filter_5")
    actualNeighbors = validationDataSet.read_neighbors(0, 10000)

    results = []
    for d in dis:
        r = []
        for i in range(min(10000, len(d))):
            r.append(d[i]['id'])
        results.append(r)

    print(actualNeighbors)
    print(f"I am in here {len(dis)}")
    for i in range(len(actualNeighbors[0])):
        if actualNeighbors[0][i] != results[0][i]:
            print(f'Failed at {i}, a : {actualNeighbors[0][i]} r: {results[0][i]}')

    with open(os.path.join(in_file_path, 'sift-128-euclidean-with-filters-updated.txt'), 'a') as file:
        for res in results:
            for r in res:
                file.write(str(r) + " ")
            file.write("\n")

    data_set_file = create_dataset_file("sift-128-euclidean-with-filters-updated", out_file_path)

    data_set_file.create_dataset("neighbors_filter_5", (len(results), len(results[0])), data=results)
    data_set_file.flush()
    data_set_file.close()


def queryTask(train_vectors, test_vectors, startIndex, endIndex, process_number, total_queries, tasks_that_are_done):
    print(f'Starting Process number : {process_number}')
    allDistances = [] * total_queries
    for i in range(total_queries):
        allDistances.insert(i, [])
    try:
        test_vectors = test_vectors[startIndex:endIndex]
        i = startIndex
        for test in test_vectors:
            distances = []
            for value in train_vectors:
                if value.apply_filter():
                    distances.append({
                        "dis": calculateL2Distance(test.vector, value.vector),
                        "id": value.id
                    })

            distances.sort(key=lambda vector: vector['dis'])
            if len(distances) > 1000:
                del distances[1000:]
            allDistances[i] = distances
            print(f"Process {process_number} queries completed: {i + 1}, queries left: {endIndex - i - 1}")
            i = i + 1
    except:
        print(
            f"Got exception while running the thread: {process_number} with startIndex: {startIndex} endIndex: {endIndex} ")
        traceback.print_exc()
    tasks_that_are_done.put(allDistances)
    print(f'Exiting Process number : {process_number}')


if __name__ == "__main__":
    main(sys.argv[1:])
