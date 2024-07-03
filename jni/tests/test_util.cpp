/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#include "test_util.h"

#include <jni.h>

#include <random>
#include <utility>

#include "faiss/Index.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetaIndexes.h"
#include "faiss/MetricType.h"
#include "faiss/impl/io.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "gmock/gmock.h"
#include "index.h"
#include "knnquery.h"
#include "methodfactory.h"
#include "params.h"
#include "space.h"

test_util::MockJNIUtil::MockJNIUtil() {
    // Set default for calls. If necessary, these can be overriden with
    // EXPECT_CALL

    // array2dJ is interpreted as a std::vector<std::vector<float>> *. Populate a
    // std::vector<float> with the elements from the 2d vector
    ON_CALL(*this, Convert2dJavaObjectArrayToCppFloatVector)
            .WillByDefault([this](JNIEnv *env, jobjectArray array2dJ, int dim) {
                std::vector<float> data;
                for (const auto &v :
                        (*reinterpret_cast<std::vector<std::vector<float>> *>(array2dJ)))
                    for (auto item : v) data.push_back(item);
                return data;
            });

    ON_CALL(*this, Convert2dJavaObjectArrayAndStoreToFloatVector)
            .WillByDefault([this](JNIEnv *env, jobjectArray array2dJ, int dim, std::vector<float>* data) {
                for (const auto &v :
                        (*reinterpret_cast<std::vector<std::vector<float>> *>(array2dJ)))
                    for (auto item : v) data->push_back(item);
            });
    ON_CALL(*this, Convert2dJavaObjectArrayAndStoreToByteVector)
            .WillByDefault([this](JNIEnv *env, jobjectArray array2dJ, int dim, std::vector<uint8_t>* data) {
                for (const auto &v :
                        (*reinterpret_cast<std::vector<std::vector<uint8_t>> *>(array2dJ)))
                    for (auto item : v) data->push_back(item);
            });


    // arrayJ is re-interpreted as std::vector<int64_t> *
    ON_CALL(*this, ConvertJavaIntArrayToCppIntVector)
            .WillByDefault([this](JNIEnv *env, jintArray arrayJ) {
                return *reinterpret_cast<std::vector<int64_t> *>(arrayJ);
            });

    // parametersJ is re-interpreted as std::unordered_map<std::string, jobject> *
    ON_CALL(*this, ConvertJavaMapToCppMap)
            .WillByDefault([this](JNIEnv *env, jobject parametersJ) {
                return *reinterpret_cast<std::unordered_map<std::string, jobject> *>(
                        parametersJ);
            });

    // objectJ is re-interpreted as int * and then dereferenced
    ON_CALL(*this, ConvertJavaObjectToCppInteger)
            .WillByDefault(
                    [this](JNIEnv *env, jobject objectJ) { return *((int *)objectJ); });

    // objectJ is re-interpreted as a std::string * and then dereferenced
    ON_CALL(*this, ConvertJavaObjectToCppString)
            .WillByDefault([this](JNIEnv *env, jobject objectJ) {
                return *((std::string *)objectJ);
            });

    // stringJ is re-interpreted as a std::string * and then dereferenced
    ON_CALL(*this, ConvertJavaStringToCppString)
            .WillByDefault([this](JNIEnv *env, jstring stringJ) {
                return *((std::string *)stringJ);
            });

    // This function should not do anything meaningful in the unit tests
    ON_CALL(*this, DeleteLocalRef)
            .WillByDefault([this](JNIEnv *env, jobject obj) {});

    // Return any value that isnt 0. This function should not return anything
    // meaningful in the unit tests
    ON_CALL(*this, FindClass)
            .WillByDefault([this](JNIEnv *env, const std::string &className) {
                return (jclass)1;
            });

    // Return any value that isnt 0. This function should not return anything
    // meaningful in the unit tests
    ON_CALL(*this, FindMethod)
            .WillByDefault(
                    [this](JNIEnv *env, const std::string &className, const std::string &methodName) {
                        return (jmethodID)1;
                    });

    // arrayJ is re-interpreted as a std::vector<uint8_t> *
    ON_CALL(*this, GetJavaBytesArrayLength)
            .WillByDefault([this](JNIEnv *env, jbyteArray arrayJ) {
                return reinterpret_cast<std::vector<uint8_t> *>(arrayJ)->size();
            });

    // arrayJ is re-interpreted as a reinterpret_cast<std::vector<uint8_t> * and
    // then the data is re-interpreted as a jbyte *
    ON_CALL(*this, GetByteArrayElements)
            .WillByDefault([this](JNIEnv *env, jbyteArray arrayJ, jboolean *isCopy) {
                return reinterpret_cast<jbyte *>(
                        reinterpret_cast<std::vector<uint8_t> *>(arrayJ)->data());
            });

    // arrayJ is re-interpreted as a std::vector<int> * and then the data is
    // re-interpreted as a jint *
    ON_CALL(*this, GetIntArrayElements)
            .WillByDefault([this](JNIEnv *env, jintArray arrayJ, jboolean *isCopy) {
                return reinterpret_cast<jint *>(
                        reinterpret_cast<std::vector<int> *>(arrayJ)->data());
            });


    // arrayJ is re-interpreted as a std::vector<int64_t> * and then the data is
    // re-interpreted as a jlong *
    ON_CALL(*this, GetLongArrayElements)
            .WillByDefault([this](JNIEnv *env, jlongArray arrayJ, jboolean *isCopy) {
                return reinterpret_cast<jlong *>(
                        reinterpret_cast<std::vector<jlong> *>(arrayJ)->data());
            });

    // arrayJ is re-interpreted as a std::vector<float> * and then the data is
    // re-interpreted as a jfloat *
    ON_CALL(*this, GetFloatArrayElements)
            .WillByDefault([this](JNIEnv *env, jfloatArray arrayJ, jboolean *isCopy) {
                return reinterpret_cast<jfloat *>(
                        reinterpret_cast<std::vector<float> *>(arrayJ)->data());
            });

    // array2dJ is re-interpreted as a std::vector<std::vector<float>> * and then
    // the size of the first element is returned
    ON_CALL(*this, GetInnerDimensionOf2dJavaFloatArray)
            .WillByDefault([this](JNIEnv *env, jobjectArray array2dJ) {
                return (*reinterpret_cast<std::vector<std::vector<float>> *>(
                        array2dJ))[0]
                        .size();
            });

    // array2dJ is re-interpreted as a std::vector<std::vector<uint8_t>> * and then
    // the size of the first element is returned
    ON_CALL(*this, GetInnerDimensionOf2dJavaByteArray)
            .WillByDefault([this](JNIEnv *env, jobjectArray array2dJ) {
                return (*reinterpret_cast<std::vector<std::vector<uint8_t>> *>(
                        array2dJ))[0]
                        .size();
            });

    // arrayJ is re-interpreted as a std::vector<float> * and the size is returned
    ON_CALL(*this, GetJavaFloatArrayLength)
            .WillByDefault([this](JNIEnv *env, jfloatArray arrayJ) {
                return reinterpret_cast<std::vector<float> *>(arrayJ)->size();
            });

    // arrayJ is re-interpreted as a std::vector<int64_t> * and then the size is
    // returned
    ON_CALL(*this, GetJavaIntArrayLength)
            .WillByDefault([this](JNIEnv *env, jintArray arrayJ) {
                return reinterpret_cast<std::vector<int64_t> *>(arrayJ)->size();
            });

    // arrayJ is re-interpreted as a std::vector<int64_t> * and then the size is
    // returned
    ON_CALL(*this, GetJavaLongArrayLength)
            .WillByDefault([this](JNIEnv *env, jlongArray arrayJ) {
                return reinterpret_cast<std::vector<int64_t> *>(arrayJ)->size();
            });

    // arrayJ is re-interpreted as a std::vector<std::vector<float>> * and then
    // the 'index' element is re-interpreted as a jobject
    ON_CALL(*this, GetObjectArrayElement)
            .WillByDefault([this](JNIEnv *env, jobjectArray arrayJ, jsize index) {
                auto vectors =
                        reinterpret_cast<std::vector<std::vector<float>> *>(arrayJ);
                return reinterpret_cast<jobject>(&((*vectors)[index]));
            });

    // create a new std::vector<uint8_t> and re-interpret it as a jbyteArray
    ON_CALL(*this, NewByteArray).WillByDefault([this](JNIEnv *env, jsize len) {
        return reinterpret_cast<jbyteArray>(new std::vector<uint8_t>());
    });

    // Create a new std::pair<int, float> with the id and distance and then
    // re-interpret it as a jobject
    ON_CALL(*this, NewObject)
            .WillByDefault([this](JNIEnv *env, jclass clazz, jmethodID methodId,
                                  int id, float distance) {
                return reinterpret_cast<jobject>(
                        new std::pair<int, float>(id, distance));
            });

    // Create a new std::vector<std::pair<int, float> and reinterpret it as a
    // jobjectArray
    ON_CALL(*this, NewObjectArray)
            .WillByDefault(
                    [this](JNIEnv *env, jsize len, jclass clazz, jobject init) {
                        return reinterpret_cast<jobjectArray>(
                                new std::vector<std::pair<int, float> *>());
                    });

    // This function should not do anything meaningful in the unit tests
    ON_CALL(*this, ReleaseByteArrayElements)
            .WillByDefault(
                    [this](JNIEnv *env, jbyteArray array, jbyte *elems, int mode) {});

    // This function should not do anything meaningful in the unit tests
    ON_CALL(*this, ReleaseFloatArrayElements)
            .WillByDefault(
                    [this](JNIEnv *env, jfloatArray array, jfloat *elems, int mode) {});

    // This function should not do anything meaningful in the unit tests
    ON_CALL(*this, ReleaseIntArrayElements)
            .WillByDefault(
                    [this](JNIEnv *env, jintArray array, jint *elems, int mode) {});

    // This function should not do anything meaningful in the unit tests
    ON_CALL(*this, ReleaseLongArrayElements)
            .WillByDefault(
                    [this](JNIEnv *env, jlongArray array, jlong *elems, int mode) {});

    // array is re-interpreted as a std::vector<uint8_t> * and then the bytes from
    // buf are copied to it
    ON_CALL(*this, SetByteArrayRegion)
            .WillByDefault([this](JNIEnv *env, jbyteArray array, jsize start,
                                  jsize len, const jbyte *buf) {
                auto byteBuffer = reinterpret_cast<std::vector<uint8_t> *>(array);
                for (int i = start; i < len; ++i) {
                    byteBuffer->push_back((uint8_t)buf[i]);
                }
            });

    // array is re-interpreted as a std::vector<std::pair<int, float> *> * and
    // then val is re-interpreted as a std::pair<int, float> * and added to the
    // vector
    ON_CALL(*this, SetObjectArrayElement)
            .WillByDefault(
                    [this](JNIEnv *env, jobjectArray array, jsize index, jobject val) {
                        reinterpret_cast<std::vector<std::pair<int, float> *> *>(array)
                                ->push_back(reinterpret_cast<std::pair<int, float> *>(val));
                    });
}

faiss::Index *test_util::FaissCreateIndex(int dim, const std::string &method,
                                          faiss::MetricType metric) {
    return faiss::index_factory(dim, method.c_str(), metric);
}

faiss::IndexBinary *test_util::FaissCreateBinaryIndex(int dim, const std::string &method) {
    return faiss::index_binary_factory(dim, method.c_str());
}

faiss::VectorIOWriter test_util::FaissGetSerializedIndex(faiss::Index *index) {
    faiss::VectorIOWriter vectorIoWriter;
    faiss::write_index(index, &vectorIoWriter);
    return vectorIoWriter;
}

faiss::VectorIOWriter test_util::FaissGetSerializedBinaryIndex(faiss::IndexBinary *index) {
    faiss::VectorIOWriter vectorIoWriter;
    faiss::write_index_binary(index, &vectorIoWriter);
    return vectorIoWriter;
}

faiss::Index *test_util::FaissLoadFromSerializedIndex(
        std::vector<uint8_t> *indexSerial) {
    faiss::VectorIOReader vectorIoReader;
    vectorIoReader.data = *indexSerial;
    return faiss::read_index(&vectorIoReader, 0);
}

faiss::IndexBinary *test_util::FaissLoadFromSerializedBinaryIndex(
        std::vector<uint8_t> *indexSerial) {
    faiss::VectorIOReader vectorIoReader;
    vectorIoReader.data = *indexSerial;
    return faiss::read_index_binary(&vectorIoReader, 0);
}

faiss::IndexIDMap test_util::FaissAddData(faiss::Index *index,
                                          std::vector<faiss::idx_t> ids,
                                          std::vector<float> dataset) {
    faiss::IndexIDMap idMap = faiss::IndexIDMap(index);
    idMap.add_with_ids(ids.size(), dataset.data(), ids.data());
    return idMap;
}

faiss::IndexBinaryIDMap test_util::FaissAddBinaryData(faiss::IndexBinary *index,
                                          std::vector<faiss::idx_t> ids,
                                          std::vector<uint8_t> dataset) {
    faiss::IndexBinaryIDMap idMap = faiss::IndexBinaryIDMap(index);
    idMap.add_with_ids(ids.size(), dataset.data(), ids.data());
    return idMap;
}

void test_util::FaissWriteIndex(faiss::Index *index,
                                const std::string &indexPath) {
    faiss::write_index(index, indexPath.c_str());
}

void test_util::FaissWriteBinaryIndex(faiss::IndexBinary *index,
                                const std::string &indexPath) {
    faiss::write_index_binary(index, indexPath.c_str());
}

faiss::Index *test_util::FaissLoadIndex(const std::string &indexPath) {
    return faiss::read_index(indexPath.c_str(), faiss::IO_FLAG_READ_ONLY);
}

faiss::IndexBinary *test_util::FaissLoadBinaryIndex(const std::string &indexPath) {
    return faiss::read_index_binary(indexPath.c_str(), faiss::IO_FLAG_READ_ONLY);
}

void test_util::FaissQueryIndex(faiss::Index *index, float *query, int k,
                                float *distances, faiss::idx_t *ids) {
    index->search(1, query, k, distances, ids);
}

void test_util::FaissTrainIndex(faiss::Index *index, faiss::idx_t n,
                                const float *x) {
    if (auto *indexIvf = dynamic_cast<faiss::IndexIVF *>(index)) {
        if (indexIvf->quantizer_trains_alone == 2) {
            test_util::FaissTrainIndex(indexIvf->quantizer, n, x);
        }
        indexIvf->make_direct_map();
    }

    if (!index->is_trained) {
        index->train(n, x);
    }
}

similarity::Index<float> *test_util::NmslibCreateIndex(
        int *ids, std::vector<std::vector<float>> dataset,
        similarity::Space<float> *space, const std::string &spaceName,
        const std::vector<std::string> &indexParameters) {
    similarity::ObjectVector objectDataset;
    for (int i = 0; i < dataset.size(); i++) {
        objectDataset.push_back(new similarity::Object(
                ids[i], -1, dataset[i].size() * sizeof(float), dataset[i].data()));
    }
    similarity::Index<float> *index =
            similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(
                    false, "hnsw", spaceName, *(space), objectDataset);
    index->CreateIndex(similarity::AnyParams(indexParameters));

    for (auto it : objectDataset) {
        delete it;
    }
    return index;
}

void test_util::NmslibWriteIndex(similarity::Index<float> *index,
                                 const std::string &indexPath) {
    index->SaveIndex(indexPath);
}

similarity::Index<float> *test_util::NmslibLoadIndex(
        const std::string &indexPath, similarity::Space<float> *space,
        const std::string &spaceName,
        const std::vector<std::string> &queryParameters) {
    similarity::ObjectVector data;
    similarity::Index<float> *index =
            similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(
                    false, "hnsw", spaceName, *space, data);

    index->LoadIndex(indexPath);
    index->SetQueryTimeParams(similarity::AnyParams(queryParameters));
    return index;
}

similarity::KNNQuery<float> *test_util::NmslibQueryIndex(
        similarity::Index<float> *index, float *query, int k, int dim,
        similarity::Space<float> *space) {
    std::unique_ptr<const similarity::Object> queryObject;
    queryObject.reset(new similarity::Object(-1, -1, dim * sizeof(float), query));

    auto *knnQuery =
            new similarity::KNNQuery<float>(*(space), queryObject.get(), k);

    index->Search(knnQuery);

    return knnQuery;
}

std::string test_util::RandomString(size_t length, const std::string &prefix,
                                    const std::string &suffix) {
    // https://stackoverflow.com/questions/440133/how-do-i-create-a-random-alpha-numeric-string-in-c
    // https://en.cppreference.com/w/cpp/numeric/random
    static constexpr auto chars =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";

    std::random_device r;
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> uniform_dist(0, std::strlen(chars) - 1);

    auto result = std::string(length, '\0');
    std::generate_n(begin(result), length,
                    [&]() { return chars[uniform_dist(e1)]; });
    return prefix + result + suffix;
}

float test_util::RandomFloat(float min, float max) {
    std::random_device r;
    std::default_random_engine e1(r());
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(e1);
}

int test_util::RandomInt(int min, int max) {
    std::random_device r;
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(e1);
}

std::vector<float> test_util::RandomVectors(int dim, int64_t numVectors, float min, float max) {
    std::vector<float> vectors(dim*numVectors);
    for (int64_t i = 0; i < dim*numVectors; i++) {
        vectors[i] = test_util::RandomFloat(min, max);
    }
    return vectors;
}

std::vector<int64_t> test_util::Range(int64_t numElements) {
    std::vector<int64_t> rangeVector(numElements);
    for (int64_t i = 0; i < numElements; i++) {
        rangeVector[i] = i;
    }
    return rangeVector;
}

size_t test_util::bits2words(uint64_t numBits) {
    return ((numBits - 1) >> 6) + 1;
}

void test_util::setBitSet(uint64_t value, jlong* array, size_t size) {
    uint64_t wordNum = value >> 6;
    if (wordNum >= size ) {
        return;
    }
    jlong bitmask = (1L << (value & 63));
    array[wordNum] |= bitmask;
}