/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.ToString;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;

import java.io.IOException;

/**
 * Concrete implementation of {@link KNNVectorValues} that returns float[] as vector and provides an abstraction over
 * {@link BinaryDocValues}, {@link ByteVectorValues}, {@link KnnFieldVectorsWriter} etc.
 */
@ToString(callSuper = true)
public class KNNByteVectorValues extends KNNVectorValues<byte[]> {
    KNNByteVectorValues(KNNVectorValuesIterator vectorValuesIterator) {
        super(vectorValuesIterator);
    }

    @Override
    public byte[] getVector() throws IOException {
        final byte[] vector = VectorValueExtractorStrategy.extractByteVector(vectorValuesIterator);
        this.dimension = vector.length;
        return vector;
    }
}
