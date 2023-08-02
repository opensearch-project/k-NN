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
import timeit
from multiprocessing import Process
from typing import cast
import numpy as np
import traceback
import h5py
import multiprocessing
import tqdm

class MyVector:

    def __init__(self, vector, id, color=None, taste=None, age=None):
        self.vector = vector
        self.id = id
        self.age = age
        self.color = color
        self.taste = taste
        self.length = 0

    def __str__(self):
        return f'Vector : {self.vector}, id : {self.id}, color: {self.color}, taste: {self.taste}, age: {self.age}\n'

    def __repr__(self):
        return f'Vector : {self.vector}, id : {self.id}, color: {self.color}, taste: {self.taste}, age: {self.age}\n'

    def setLength(self, leng):
        self.length = leng

class HDF5DataSet:

    def __init__(self, file_path, key):
        self.file_name = file_path
        self.file = h5py.File(self.file_name)
        self.key = key
        self.data = cast(h5py.Dataset, self.file[key])
        print(f'Keys in the file are {self.file.keys()}')

    def read_as_vector(self, start, end=None):
        if end is None:
            end = self.data.len()
        values = cast(np.ndarray, self.data[start:end])
        vectors = []
        i = 0
        for value in values:
            vector = MyVector(value, i)
            vectors.append(vector)
            i = i + 1
        return vectors

    def read(self, start, end=None):
        if end is None:
            end = self.data.len()
        return cast(np.ndarray, self.data[start:end])

    def size(self):
        return self.data.len()


def calculateL2Distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def create_dataset_file(output_file) -> h5py.File:
    if os.path.isfile(output_file):
        os.remove(output_file)
    else:
        print(f"Creating the output file at {output_file}")
    data_set_w_filtering = h5py.File(output_file, 'a')

    return data_set_w_filtering


def query_task_only_vector_search(train_vectors, trained_dis_vector, hdf5Data_test, startIndex, endIndex, process_number, total_queries,
                                  tasks_that_are_done):
    print(f'Starting Process number : {process_number}')
    allDistances = [] * total_queries
    k = 100
    for i in range(total_queries):
        allDistances.insert(i, [])
    try:
        test_vectors = hdf5Data_test.read(startIndex, endIndex)
        i = startIndex
        for test in tqdm.tqdm(test_vectors):
            distances = []
            starttime = timeit.default_timer()
            # reference: https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/bruteforce/module.py#L83-L85
            # not computing query lengths as they are constant and will not impact the order
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab  # noqa
            dists = trained_dis_vector - 2 * np.dot(train_vectors, test)
            nearest_indices = np.argpartition(dists, k)[:k]
            for idx in nearest_indices:
                distances.append({
                    "dis": 0,
                    "id": idx
                })
            print("Total time to taken for getting top k neighbors is :", timeit.default_timer() - starttime)
            allDistances[i] = distances
            i = i + 1
    except:
        print(
            f"Got exception while running the thread: {process_number} with startIndex: {startIndex} endIndex: {endIndex} ")
        traceback.print_exc()
    tasks_that_are_done.put(allDistances)
    print(f'Exiting Process number : {process_number}')


def generate_ground_truth(in_file_path, out_file_path, query_clients, corpus_size=None):
    data_set_file = create_dataset_file(out_file_path)
    total_clients = query_clients
    hdf5Data_train = HDF5DataSet(in_file_path, "train")

    if corpus_size is None:
        corpus_size = hdf5Data_train.size()

    train_vectors = hdf5Data_train.read(0, corpus_size)
    print(f'Train vector size: {len(train_vectors)}')

    print("Precomputing distance for the corpus")
    trained_dis_vector = []
    for vector in tqdm.tqdm(train_vectors):
        lens = (vector**2).sum(-1)
        trained_dis_vector.append(lens)

    trained_dis_vector = np.asarray(trained_dis_vector)

    print("Distance precomputation completed")

    hdf5Data_test = HDF5DataSet(in_file_path, "test")
    total_queries = hdf5Data_test.size()
    dis = [] * total_queries

    for i in range(total_queries):
        dis.insert(i, [])

    queries_per_client = int(total_queries / total_clients)
    if queries_per_client == 0:
        queries_per_client = total_queries

    processes = []
    tasks_that_are_done = multiprocessing.Queue()
    for client in range(total_clients):
        start_index = int(client * queries_per_client)
        if start_index + queries_per_client <= total_queries:
            end_index = int(start_index + queries_per_client)
        else:
            end_index = total_queries - start_index

        print(f'Client {client} will process from start Index: {start_index}, end Index: {end_index}')
        p = Process(target=query_task_only_vector_search, args=(
            train_vectors, trained_dis_vector, hdf5Data_test, start_index, end_index, client, total_queries, tasks_that_are_done))
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
    print("All processes completed now adding data in file")
    results = []
    ids_distance = []
    for d in dis:
        r = []
        ids_dis = []
        for i in range(min(1000, len(d))):
            r.append(d[i]['id'])
            ids_dis.append(d[i]['dis'])
        results.append(r)
        ids_distance.append(ids_dis)

    data_set_file.create_dataset("neighbors", (len(results), len(results[0])), data=results)
    data_set_file.create_dataset("distances", (len(results), len(results[0])), data=ids_distance)
    data_set_file.flush()
    data_set_file.close()
    print(f"The ground truth file is generated and added at : {out_file_path}")


def main(argv):
    opts, args = getopt.getopt(argv, "", ["input_file=", "output_file=", "corpus_size=", "query_client="])
    print(f'Options provided are: {opts}')
    print(f'Arguments provided are: {args}')
    inputfile = None
    outputfile = None
    corpus_size = None
    # given that distance calculations are going to use most of the processors so its better to use less number of cores.
    query_clients = min(2, multiprocessing.cpu_count())
    for opt, arg in opts:
        if opt == '-h':
            print('--input_file <inputfile> --output_file <outputfile>')
            sys.exit()
        elif opt in "--input_file":
            inputfile = arg
        elif opt in "--output_file":
            outputfile = arg
        elif opt in "--corpus_size":
            corpus_size = int(arg)
        elif opt in "--query_client":
            query_clients = int(arg)

    if inputfile is None:
        print(f"The input file is not provided.")
        sys.exit()

    if outputfile is None:
        print(f"The output file is not provided.")
        sys.exit()
    starttime = timeit.default_timer()
    generate_ground_truth(in_file_path=inputfile, out_file_path=outputfile, corpus_size=corpus_size, query_clients=query_clients)
    print("Total time to generate the dataset is :", timeit.default_timer() - starttime)


if __name__ == "__main__":
    main(sys.argv[1:])
