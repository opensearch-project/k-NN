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

package org.opensearch.knn.index.codec.KNN990Codec;

import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

/**
 * FP16 variant of {@link NativeEngines990KnnVectorsFormat}, used for {@code half_float} fields
 * with the flat encoder.
 */
public class NativeEngines990HalfFloatKnnVectorsFormat extends NativeEngines990KnnVectorsFormat {

    private static final String FORMAT_NAME = "NativeEngines990HalfFloatKnnVectorsFormat";

    public NativeEngines990HalfFloatKnnVectorsFormat() {
        super(FORMAT_NAME, KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE, new NativeIndexBuildStrategyFactory(), true);
    }

    public NativeEngines990HalfFloatKnnVectorsFormat(
        int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(FORMAT_NAME, approximateThreshold, nativeIndexBuildStrategyFactory, true);
    }
}
