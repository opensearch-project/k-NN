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

#ifndef KNNPLUGIN_JNI_INCLUDE_PARAMETER_UTILS_H_
#define KNNPLUGIN_JNI_INCLUDE_PARAMETER_UTILS_H_

#include <stdexcept>
#include <string>

namespace knn_jni {
namespace util {

struct ParameterCheck {
  template<typename PtrType>
  static PtrType *require_non_null(PtrType *ptr, const char *parameter_name) {
    if (ptr == nullptr) {
      throw std::invalid_argument(std::string("Parameter [") + parameter_name + "] should not be null.");
    }
    return ptr;
  }

 private:
  ParameterCheck() = default;
};  // class ParameterCheck



}
}  // namespace knn_jni

#endif //KNNPLUGIN_JNI_INCLUDE_PARAMETER_UTILS_H_
