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

#include "native_engines_stream_support.h"
#include "utils.h"  // This is from NMSLIB
#include "parameter_utils.h"

namespace knn_jni {
namespace stream {

/**
 * NmslibIOReader implementation delegating NativeEngineIndexInputMediator to read bytes.
 */
class NmslibOpenSearchIOReader final : public similarity::NmslibIOReader {
 public:
  explicit NmslibOpenSearchIOReader(NativeEngineIndexInputMediator *_mediator)
      : similarity::NmslibIOReader(),
        mediator(knn_jni::util::ParameterCheck::require_non_null(_mediator, "mediator")) {
  }

  void read(char *bytes, size_t len) final {
    if (len > 0) {
      // Mediator calls IndexInput, then copy read bytes to `ptr`.
      mediator->copyBytes(len, (uint8_t *) bytes);
    }
  }

  size_t remainingBytes() final {
    return mediator->remainingBytes();
  }

 private:
  NativeEngineIndexInputMediator *mediator;
};  // class NmslibOpenSearchIOReader


class NmslibOpenSearchIOWriter final : public similarity::NmslibIOWriter {
 public:
  explicit NmslibOpenSearchIOWriter(NativeEngineIndexOutputMediator *_mediator)
      : similarity::NmslibIOWriter(),
        mediator(knn_jni::util::ParameterCheck::require_non_null(_mediator, "mediator")) {
  }

  void write(char *bytes, size_t len) final {
    if (len > 0) {
      mediator->writeBytes((uint8_t *) bytes, len);
    }
  }

  void flush() final {
    mediator->flush();
  }

 private:
  NativeEngineIndexOutputMediator *mediator;
};  // class NmslibOpenSearchIOWriter


}
}

#endif //OPENSEARCH_KNN_JNI_NMSLIB_STREAM_SUPPORT_H
