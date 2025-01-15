/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search.distance;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.storage.Storage;

import java.io.IOException;

public abstract class DistanceComputer {
    public static DistanceComputer createDistanceFunctionFromFlatVector(
        Storage flatVectors,
        PartialLoadingSearchParameters searchParameters
    ) {
        if (searchParameters.getFloatQueryVector() != null) {
            return new FloatVectorDistanceComputer(
                searchParameters.getSpaceType(),
                searchParameters.getFloatQueryVector(),
                flatVectors,
                searchParameters.getIndexInput()
            );
        } else if (searchParameters.getByteQueryVector() != null) {
            // TODO : KDY
            return null;
        } else {
            throw new IllegalArgumentException("Distance function needs at least one query vector. Both float[] and byte[] were null.");
        }
    }

    public static boolean needReverseScore(SpaceType spaceType) {
        return spaceType == SpaceType.INNER_PRODUCT;
    }

    public abstract float compute(long index) throws IOException;
}
