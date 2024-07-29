/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.ToString;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;

import java.io.IOException;

/**
 * An abstract class to iterate over KNNVectors, as KNNVectors are stored as different representation like
 * {@link BinaryDocValues}, {@link FloatVectorValues}, {@link ByteVectorValues}, {@link KnnFieldVectorsWriter} etc.
 * @param <T>
 */
@ToString
public abstract class KNNVectorValues<T> {

    protected final KNNVectorValuesIterator vectorValuesIterator;
    protected int dimension;

    protected KNNVectorValues(final KNNVectorValuesIterator vectorValuesIterator) {
        this.vectorValuesIterator = vectorValuesIterator;
    }

    /**
     * Return a vector reference. If you are adding this address in a List/Map ensure that you are copying the vector first.
     * This is to ensure that we keep the heap and latency in check by reducing the copies of vectors.
     *
     * @return T an array of byte[], float[]
     * @throws IOException if we are not able to get the vector
     */
    public abstract T getVector() throws IOException;

    /**
     * Dimension of vector is returned. Do call getVector function first before calling this function otherwise you will get 0 value.
     * @return int
     */
    public int dimension() {
        assert docId() != -1 && dimension != 0 : "Cannot get dimension before we retrieve a vector from KNNVectorValues";
        return dimension;
    }

    /**
     * Returns the total live docs for KNNVectorValues.
     * @return long
     */
    public long totalLiveDocs() {
        return vectorValuesIterator.liveDocs();
    }

    /**
     * Returns the current docId where the iterator is pointing to.
     * @return int
     */
    public int docId() {
        return vectorValuesIterator.docId();
    }

    /**
     * Advances to a specific docId. Ensure that the passed docId is greater than current docId where Iterator is
     * pointing to, otherwise
     * {@link IOException} will be thrown
     * @return int
     * @throws IOException if we are not able to move to the passed docId.
     */
    public int advance(int docId) throws IOException {
        return vectorValuesIterator.advance(docId);
    }

    /**
     * Move to nextDocId.
     * @return int
     * @throws IOException if we cannot move to next docId
     */
    public int nextDoc() throws IOException {
        return vectorValuesIterator.nextDoc();
    }

}
