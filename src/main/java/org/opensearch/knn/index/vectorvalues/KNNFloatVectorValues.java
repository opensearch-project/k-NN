/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FloatVectorValues;

import java.io.IOException;

/**
 * Concrete implementation of {@link KNNVectorValues} that returns float[] as vector and provides an abstraction over
 * {@link BinaryDocValues}, {@link FloatVectorValues}, {@link KnnFieldVectorsWriter} etc.
 */
public class KNNFloatVectorValues extends KNNVectorValues<float[]> {
    KNNFloatVectorValues(final KNNVectorValuesIterator vectorValuesIterator) {
        super(vectorValuesIterator);
    }

    @Override
    public float[] getVector() throws IOException {
        final float[] vector = VectorValueExtractorStrategy.extractFloatVector(vectorValuesIterator);
        this.dimension = vector.length;
        return vector;
    }
}
