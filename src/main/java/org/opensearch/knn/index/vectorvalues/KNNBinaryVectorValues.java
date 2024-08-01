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
 * Concrete implementation of {@link KNNVectorValues} that returns byte[] as vector where binary vector is stored and
 * provides an abstraction over {@link BinaryDocValues}, {@link ByteVectorValues}, {@link KnnFieldVectorsWriter} etc.
 */
@ToString(callSuper = true)
public class KNNBinaryVectorValues extends KNNVectorValues<byte[]> {
    KNNBinaryVectorValues(KNNVectorValuesIterator vectorValuesIterator) {
        super(vectorValuesIterator);
    }

    @Override
    public byte[] getVector() throws IOException {
        final byte[] vector = VectorValueExtractorStrategy.extractBinaryVector(vectorValuesIterator);
        this.dimension = vector.length;
        return vector;
    }

    /**
     * Binary Vector values gets stored as byte[], hence for dimension of the binary vector we have to multiply the
     * byte[] size with {@link Byte#SIZE}
     * @return int
     */
    @Override
    public int dimension() {
        return super.dimension() * Byte.SIZE;
    }
}
