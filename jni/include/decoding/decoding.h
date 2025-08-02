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

#ifndef OPENSEARCH_KNN_DECODING_H
#define OPENSEARCH_KNN_DECODING_H

#pragma once

#if defined(__aarch64__) && defined(KNN_HAVE_ARM_FP16)
  #include "arm_decoding.h"
#elif defined(__x86_64__) && (defined(KNN_HAVE_AVX512) || defined(KNN_HAVE_F16C))
  #include "x86_decoding.h"
#else
  #include "default_decoding.h"
#endif

#endif