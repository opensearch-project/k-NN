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

#ifndef OPENSEARCH_KNN_JNI_FAISS_STREAM_SUPPORT_H
#define OPENSEARCH_KNN_JNI_FAISS_STREAM_SUPPORT_H

#include "faiss/impl/io.h"
#include "jni_util.h"
#include "native_engines_stream_support.h"
#include "parameter_utils.h"

#include <jni.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace knn_jni {
namespace stream {



/**
 * A glue component inheriting IOReader to be passed down to Faiss library.
 * This will then indirectly call the mediator component and eventually read required bytes from Lucene's IndexInput.
 */
class FaissOpenSearchIOReader final : public faiss::IOReader {
public:
    explicit FaissOpenSearchIOReader(NativeEngineIndexInputMediator* _mediator)
        : faiss::IOReader(),
          mediator(knn_jni::util::ParameterCheck::require_non_null(_mediator, "mediator")) {

        name = "FaissOpenSearchIOReader";

        // Step 1: Access env and readStream from mediator
        JNIEnv* env = mediator->getEnv();
        jobject readStream = mediator->getJavaObject();

        // Step 2: Get isFloatVector() method to determine vector type
        jclass streamClass = env->GetObjectClass(readStream);
        jmethodID isFloatMid = env->GetMethodID(streamClass, "isFloatVector", "()Z");
        if (!isFloatMid) {
            throw std::runtime_error("Failed to find method isFloatVector()");
        }
        jboolean isFloat = env->CallBooleanMethod(readStream, isFloatMid);

        // Step 3: Get VectorReader from getFullPrecisionVectors()
        jmethodID getVectorsMid = env->GetMethodID(
            streamClass,
            "getFullPrecisionVectors",
            "()Lorg/opensearch/knn/index/store/VectorReader;"
        );
        if (!getVectorsMid) {
            throw std::runtime_error("Failed to find method getFullPrecisionVectors()");
        }

        jobject vectorReader = env->CallObjectMethod(readStream, getVectorsMid);
        if (env->ExceptionCheck() || vectorReader == nullptr) {
            throw std::runtime_error("getFullPrecisionVectors() returned null");
        }

        // Step 4: Create global reference
        vectorReaderGlobalRef = env->NewGlobalRef(vectorReader);
        if (vectorReaderGlobalRef == nullptr) {
            throw std::runtime_error("Failed to create global reference for VectorReader");
        }

        // Step 5: Cache getNextVector method based on type
        jclass readerClass = env->GetObjectClass(vectorReaderGlobalRef);
        if (isFloat) {
            nextMid = env->GetMethodID(readerClass, "nextFloatVector", "()[F");
            if (!nextMid) {
                throw std::runtime_error("Failed to find method nextFloatVector()");
            }
            vectorDataType = VectorType::FLOAT;
        } else {
            nextMid = env->GetMethodID(readerClass, "nextByteVector", "()[B");
            if (!nextMid) {
                throw std::runtime_error("Failed to find method nextByteVector()");
            }
            vectorDataType = VectorType::BYTE;
        }

        cachedEnv = env;  // Save env for destructor
    }

    ~FaissOpenSearchIOReader() override {
        if (vectorReaderGlobalRef && cachedEnv) {
            cachedEnv->DeleteGlobalRef(vectorReaderGlobalRef);
        }
    }

    size_t operator()(void* ptr, size_t size, size_t nitems) override {
        const auto bytes = size * nitems;
        mediator->copyBytes(bytes, static_cast<uint8_t*>(ptr));
        return nitems;
    }

    bool copy(void* dest, int expectedByteSize) override {
        JNIEnv* env = cachedEnv;

        if (vectorDataType == VectorType::FLOAT) {
            jfloatArray vector = (jfloatArray) env->CallObjectMethod(vectorReaderGlobalRef, nextMid);
            if (env->ExceptionCheck() || vector == nullptr) {
                return false;
            }

            jsize length = env->GetArrayLength(vector);
            jfloat* elems = env->GetFloatArrayElements(vector, nullptr);

            JNIReleaseElements release_elems([=]() {
                env->ReleaseFloatArrayElements(vector, elems, JNI_ABORT);
            });

            int vectorByteSize = sizeof(float) * length;
            if (vectorByteSize != expectedByteSize) {
                return false;
            }

            std::memcpy(dest, elems, vectorByteSize);
            return true;

        } else if (vectorDataType == VectorType::BYTE) {
            jbyteArray vector = (jbyteArray) env->CallObjectMethod(vectorReaderGlobalRef, nextMid);
            if (env->ExceptionCheck() || vector == nullptr) {
                return false;
            }

            jsize length = env->GetArrayLength(vector);
            jbyte* elems = env->GetByteArrayElements(vector, nullptr);

            JNIReleaseElements release_elems([=]() {
                env->ReleaseByteArrayElements(vector, elems, JNI_ABORT);
            });

            int vectorByteSize = sizeof(float) * length;
            if (vectorByteSize != expectedByteSize) {
                return false;
            }

            float* floatDest = static_cast<float*>(dest);
            for (int i = 0; i < length; ++i) {
                floatDest[i] = static_cast<float>(elems[i]);  // Convert byte to float
            }

            return true;
        }

        return false;
    }

private:
    enum class VectorType { FLOAT, BYTE };

    NativeEngineIndexInputMediator* mediator;
    jobject vectorReaderGlobalRef = nullptr;
    jmethodID nextMid = nullptr;
    JNIEnv* cachedEnv = nullptr;
    VectorType vectorDataType;
}; // class FaissOpenSearchIOReader


/**
 * A glue component inheriting IOWriter to delegate IO processing down to the given
 * mediator. The mediator is expected to do write bytes via the provided Lucene's IndexOutput.
 */
class FaissOpenSearchIOWriter final : public faiss::IOWriter {
 public:
  explicit FaissOpenSearchIOWriter(NativeEngineIndexOutputMediator *_mediator)
      : faiss::IOWriter(),
        mediator(knn_jni::util::ParameterCheck::require_non_null(_mediator, "mediator")) {
    name = "FaissOpenSearchIOWriter";
  }

  size_t operator()(const void *ptr, size_t size, size_t nitems) final {
    const auto writeBytes = size * nitems;
    if (writeBytes > 0) {
      mediator->writeBytes(reinterpret_cast<const uint8_t *>(ptr), writeBytes);
    }
    return nitems;
  }

  // return a file number that can be memory-mapped
  int filedescriptor() final {
    throw std::runtime_error("filedescriptor() is not supported in FaissOpenSearchIOWriter.");
  }

  void flush() {
    mediator->flush();
  }

 private:
  NativeEngineIndexOutputMediator *mediator;
};  // class FaissOpenSearchIOWriter



}
}

#endif //OPENSEARCH_KNN_JNI_FAISS_STREAM_SUPPORT_H
