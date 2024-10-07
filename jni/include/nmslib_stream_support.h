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

#ifndef OPENSEARCH_KNN_JNI_NMSLIB_STREAM_SUPPORT_H
#define OPENSEARCH_KNN_JNI_NMSLIB_STREAM_SUPPORT_H

#include "jni_util.h"
#include "native_engines_stream_support.h"

#include <jni.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace knn_jni {
namespace stream {



/**
 * TODO : KDY
 */
class NmslibMediatorInputStreamBuffer final : public std::streambuf {
 public:
  explicit NmslibMediatorInputStreamBuffer(NativeEngineIndexInputMediator *_mediator)
      : std::streambuf(),
        mediator(_mediator) {
  }

 protected:
  std::streamsize xsgetn(std::streambuf::char_type *destination, std::streamsize count) final {
    if (count > 0) {
      mediator->copyBytes(count, (uint8_t *) destination);
    }
    return count;
  }

 private:
  NativeEngineIndexInputMediator *mediator;
};  // NmslibMediatorInputStreamBuffer



}
}

#endif //OPENSEARCH_KNN_JNI_NMSLIB_STREAM_SUPPORT_H
