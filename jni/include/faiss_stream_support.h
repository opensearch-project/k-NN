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
  explicit FaissOpenSearchIOReader(NativeEngineIndexInputMediator *_mediator)
      : faiss::IOReader(),
        mediator(_mediator) {
    name = "FaissOpenSearchIOReader";
  }

  size_t operator()(void *ptr, size_t size, size_t nitems) final {
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
  NativeEngineIndexInputMediator *mediator;
};  // class FaissOpenSearchIOReader



}
}

#endif //OPENSEARCH_KNN_JNI_FAISS_STREAM_SUPPORT_H
