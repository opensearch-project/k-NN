/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

/**
 * Utility class used to resolve the space type of a KNNMethodConfigContext
 */
public class SpaceTypeResolver {
    /**
     * Resolves the engine, given the context
     *
     * @param knnMethodConfigContext context to use for resolution
     * @return engine to use for the knn method
     */
    public static SpaceType resolveSpaceType(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext == null) {
            return getDefault(knnMethodConfigContext);
        }
        KNNMethodContext knnMethodContext = knnMethodConfigContext.getKnnMethodContext();
        if (knnMethodContext == null) {
            return getDefault(knnMethodConfigContext);
        }
        return knnMethodContext.getSpaceType().orElse(getDefault(knnMethodConfigContext));
    }

    private static SpaceType getDefault(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext != null && knnMethodConfigContext.getVectorDataType() == VectorDataType.BINARY) {
            return SpaceType.HAMMING;
        }
        return SpaceType.L2;
    }
}
