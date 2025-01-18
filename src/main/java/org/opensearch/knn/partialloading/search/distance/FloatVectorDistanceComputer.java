/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search.distance;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.KNNVectorDistanceFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.partialloading.KdyPerfCheck;

import java.io.IOException;

/**
 * Calculates the distance between a stored float vector and a float query vector.
 */
public class FloatVectorDistanceComputer extends DistanceComputer {
    private final float[] queryVector;
    private final float[] floatBuffer;
    private final KNNVectorDistanceFunction distanceFunction;
    private final IndexInput indexInput;
    private final int oneVectorByteSize;

    public FloatVectorDistanceComputer(SpaceType spaceType, float[] queryVector, IndexInput indexInput) {
        this.queryVector = queryVector;
        final int dimension = queryVector.length;
        floatBuffer = new float[dimension];
        oneVectorByteSize = Float.BYTES * dimension;
        this.distanceFunction = spaceType.getKnnVectorDistanceFunction();
        this.indexInput = indexInput;
    }

    public float compute(long vectorId) throws IOException {
        KdyPerfCheck.incVectorVisit();

        // Load float vector
        indexInput.seek(vectorId * oneVectorByteSize);
        indexInput.readFloats(floatBuffer, 0, floatBuffer.length);

        // Calculate distance
        return distanceFunction.distance(queryVector, floatBuffer);
    }
}
