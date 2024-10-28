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

#ifndef KNNPLUGIN_JNI_INCLUDE_MEMORY_UTIL_H_
#define KNNPLUGIN_JNI_INCLUDE_MEMORY_UTIL_H_

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __declspec(restrict)
#else
#define RESTRICT
#endif

#endif //KNNPLUGIN_JNI_INCLUDE_MEMORY_UTIL_H_
