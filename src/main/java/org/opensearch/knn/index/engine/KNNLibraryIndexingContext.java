/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;
import org.opensearch.knn.index.mapper.VectorTransformer;
import org.opensearch.knn.index.mapper.VectorValidator;

import java.util.Map;

/**
 * Context a library gives to build one of its indices
 */
public interface KNNLibraryIndexingContext {
    /**
     * Get map of parameters that get passed to the library to build the index
     *
     * @return Map of parameters
     */
    Map<String, Object> getLibraryParameters();

    /**
     * Get map of parameters that get passed to the quantization framework
     *
     * @return Map of parameters
     */
    QuantizationConfig getQuantizationConfig();

    /**
     *
     * @return Get the vector validator
     */
    VectorValidator getVectorValidator();

    /**
     *
     * @return Get the per dimension validator
     */
    PerDimensionValidator getPerDimensionValidator();

    /**
     *
     * @return Get the per dimension processor
     */
    PerDimensionProcessor getPerDimensionProcessor();

    /**
     * Get the vector transformer that will be used to transform the vector before indexing.
     * This will be applied at vector level once entire vector is parsed and validated.
     *
     * @return VectorTransformer
     */
    VectorTransformer getVectorTransformer();
}
