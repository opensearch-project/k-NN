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

package org.opensearch.knn.index.query.refine;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

/**
 * Context required to refine with higher precision results from a k-NN query
 *
 **/
@AllArgsConstructor
@Getter
public class RefineContext {
    private final String field;
    private final SpaceType spaceType;
    private final float[] queryVector;

    /**
     * Refine score using index vector
     *
     * @param indexVector vector to be rescored against query vector
     * @return new, higher precision score
     */
    public float refine(float[] indexVector) {
        return spaceType.getVectorSimilarityFunction().compare(queryVector, indexVector);
    }

    /**
     * Get vector values from leaf reader context
     *
     * @param leafReaderContext leaf reader context
     * @return KNNVectorValues
     */
    public KNNVectorValues<float[]> getKNNVectorValues(LeafReaderContext leafReaderContext) throws IOException {
        return KNNVectorValuesFactory.getFloatVectorValues(leafReaderContext, field);
    }
}
