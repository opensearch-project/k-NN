#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

echo "Executing LinuxScript"

git submodule update --init -- jni/external/faiss

# Temporarily replace the intrinsic '_mm_loadu_si64' with '_mm_loadl_epi64' until centOS7 is deprecated.
# centOS7 only supports gcc version upto 8.x. But, the intrinsic '_mm_loadu_si64' requires gcc version of minimum 9.x.
# So, replacing it with an equivalent intrinsic.

sed -i -e 's/_mm_loadu_si64/_mm_loadl_epi64/g' jni/external/faiss/faiss/impl/code_distance/code_distance-avx2.h