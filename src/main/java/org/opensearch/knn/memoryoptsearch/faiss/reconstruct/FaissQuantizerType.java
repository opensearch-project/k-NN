/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

// Refer to https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ScalarQuantizer.h#L27
public enum FaissQuantizerType {
    QT_8BIT,
    QT_4BIT,
    QT_8BIT_UNIFORM,
    QT_4BIT_UNIFORM,
    QT_FP16,
    QT_8BIT_DIRECT,
    QT_6BIT,
    QT_BF16,
    QT_8BIT_DIRECT_SIGNED
}
