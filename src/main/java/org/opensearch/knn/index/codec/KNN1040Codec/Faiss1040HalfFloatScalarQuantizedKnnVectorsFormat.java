/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

/**
 * FP16 variant of {@link Faiss1040ScalarQuantizedKnnVectorsFormat}, used when a {@code sq, bits:1}
 * Faiss HNSW field is declared {@code half_float}.
 */
public class Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat extends Faiss1040ScalarQuantizedKnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat";

    public Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat() {
        this(new NativeIndexBuildStrategyFactory());
    }

    public Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat(final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory) {
        super(FORMAT_NAME, nativeIndexBuildStrategyFactory, VectorDataType.HALF_FLOAT);
    }
}
