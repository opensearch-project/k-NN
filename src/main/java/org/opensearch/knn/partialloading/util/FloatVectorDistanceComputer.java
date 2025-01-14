/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.util;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.KNNVectorDistanceFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.partialloading.storage.Storage;

import java.io.IOException;

public class FloatVectorDistanceComputer extends DistanceComputer {
    private final float[] queryVector;
    private final float[] floatBuffer;
    private final Storage codes;
    private final KNNVectorDistanceFunction distanceFunction;
    private final IndexInput indexInput;
    private final int oneVectorByteSize;

    public FloatVectorDistanceComputer(SpaceType spaceType, float[] queryVector, Storage codes, IndexInput indexInput) {
        this.queryVector = queryVector;
        final int dimension = queryVector.length;
        floatBuffer = new float[dimension];
        oneVectorByteSize = Float.BYTES * dimension;
        this.codes = codes;
        this.distanceFunction = spaceType.getKnnVectorDistanceFunction();
        this.indexInput = indexInput;
    }

    public float compute(long index) throws IOException {
        indexInput.seek(codes.getBaseOffset() + index * oneVectorByteSize);
        indexInput.readFloats(floatBuffer, 0, floatBuffer.length);
        return distanceFunction.distance(queryVector, floatBuffer);
    }
}
