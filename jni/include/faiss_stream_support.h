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

#ifndef OPENSEARCH_KNN_JNI_STREAM_SUPPORT_H
#define OPENSEARCH_KNN_JNI_STREAM_SUPPORT_H

#include "faiss/impl/io.h"

#include <jni.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace knn_jni { namespace stream {



/**
 * This class contains Java IndexInputWithBuffer reference and calls its API to copy required bytes into a read buffer.
 */
class NativeEngineIndexInputMediator {
 public:
  // Expect IndexInputWithBuffer is given as `_indexInput`.
  NativeEngineIndexInputMediator(JNIEnv * _env, jobject _indexInput)
    : env(_env),
      indexInput(_indexInput),
      bufferArray((jbyteArray) (_env->GetObjectField(_indexInput, getBufferFieldId(_env)))),
      copyBytesMethod(getCopyBytesMethod(_env)) {
  }

  void copyBytes(int32_t nbytes, uint8_t* destination) {
      while (nbytes > 0) {
          // Call `copyBytes` to read bytes as many as possible.
          const auto readBytes =
            env->CallIntMethod(indexInput, copyBytesMethod, nbytes);

          // === Critical Section Start ===

          // Get primitive array pointer, no copy is happening in OpenJDK.
          jbyte* primitiveArray =
            (jbyte*) env->GetPrimitiveArrayCritical(bufferArray, NULL);

          // Copy Java bytes to C++ destination address.
          std::memcpy(destination, primitiveArray, readBytes);

          // Release the acquired primitive array pointer.
          // JNI_ABORT tells JVM to directly free memory without copying back to Java byte[].
          // Since we're merely copying data, we don't need to copying back.
          env->ReleasePrimitiveArrayCritical(bufferArray, primitiveArray, JNI_ABORT);

          // === Critical Section End ===

          destination += readBytes;
          nbytes -= readBytes;
      }  // End while
  }

  private:
    static jclass getIndexInputWithBufferClass(JNIEnv * env) {
       static jclass INDEX_INPUT_WITH_BUFFER_CLASS =
                  env->FindClass("org/opensearch/knn/index/util/IndexInputWithBuffer");
       return INDEX_INPUT_WITH_BUFFER_CLASS;
    }

    static jmethodID getCopyBytesMethod(JNIEnv *env) {
        static jmethodID COPY_METHOD_ID =
            env->GetMethodID(getIndexInputWithBufferClass(env), "copyBytes", "(J)I");
        return COPY_METHOD_ID;
    }

    static jfieldID getBufferFieldId(JNIEnv *env) {
        static jfieldID BUFFER_FIELD_ID = env->GetFieldID(getIndexInputWithBufferClass(env), "buffer", "[B");
        return BUFFER_FIELD_ID;
    }

    JNIEnv * env;

    // `IndexInputWithBuffer` instance having `IndexInput` instance obtained from `Directory` for reading.
    jobject indexInput;
    jbyteArray bufferArray;
    jmethodID copyBytesMethod;
}; // class NativeEngineIndexInputMediator



/**
 * A glue component inheriting IOReader to be passed down to Faiss library.
 * This will then indirectly call the mediator component and eventually read required bytes from Lucene's IndexInput.
 */
class FaissOpenSearchIOReader final : public faiss::IOReader {
public:
    explicit FaissOpenSearchIOReader(NativeEngineIndexInputMediator* _mediator)
      : faiss::IOReader(),
        mediator(_mediator) {
        name = "FaissOpenSearchIOReader";
    }

    size_t operator()(void* ptr, size_t size, size_t nitems) final {
        const auto readBytes = size * nitems;
        if (readBytes > 0) {
            // Mediator calls IndexInput, then copy read bytes to `ptr`.
            mediator->copyBytes(readBytes, (uint8_t *) ptr);
        }
        return nitems;
    }

    int filedescriptor() final {
        throw std::runtime_error("filedescriptor() is not supported in FaissOpenSearchIOReader.");
    }

private:
    NativeEngineIndexInputMediator* mediator;
};  // class FaissOpenSearchIOReader



}
}

#endif //OPENSEARCH_KNN_JNI_STREAM_SUPPORT_H
