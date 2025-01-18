/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search.distance;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.storage.Storage;

import java.io.IOException;

/**
 * A distance computer that calculates the distance between a query vector and a vector at a given index.
 * It will locate the vector using the provided index, then computes the distance to the query vector. If the returned distance is larger,
 * the search framework will consider the vector to be farther from the query vector.
 * For metrics where a greater distance indicates closer proximity to the query vector, the distance must be negated before returning.
 * Additionally, the metric should be added to {@link DistanceComputer#needReverseScore(SpaceType)}.
 */
public abstract class DistanceComputer {
    public static DistanceComputer createDistanceFunctionFromFlatVector(
        Storage flatVectors, PartialLoadingSearchParameters searchParameters
    ) {
        try {
            IndexInput vectorIndexInput =
                searchParameters.getIndexInput().slice("FaissDistanceComputer", flatVectors.getBaseOffset(), flatVectors.getSectionSize());

            if (searchParameters.getFloatQueryVector() != null) {
                return new FloatVectorDistanceComputer(searchParameters.getSpaceType(),
                                                       searchParameters.getFloatQueryVector(),
                                                       vectorIndexInput
                );
            } else if (searchParameters.getByteQueryVector() != null) {
                throw new UnsupportedOperationException("Partial loading does not support byte query yet.");
            } else {
                throw new IllegalArgumentException("Distance function needs at least one query vector. Both float[] and byte[] were null.");
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Determines whether the result distance must be negated.
     * Some metrics, such as inner product, use a greater distance value to indicate closer proximity to the query vector.
     * @param spaceType Space type.
     * @return True if its distance must be reversed, False otherwise.
     */
    public static boolean needReverseScore(SpaceType spaceType) {
        return spaceType == SpaceType.INNER_PRODUCT;
    }

    /**
     * Calculates the distance between a vector with the given ID and the query vector.
     * A greater distance value indicates that the vector is farther from the query vector. If the distance definition for the underlying
     * space type is the opposite, the method should return a negated distance instead.
     *
     * @param vectorId Physical vector id in FAISS index to be located by this computer.
     * @return Distance value. Greater it is, father it is to a query vector.
     * @throws IOException
     */
    public abstract float compute(long vectorId) throws IOException;
}
