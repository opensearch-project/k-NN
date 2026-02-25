/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.Getter;
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

    @Getter
    protected final KNNVectorValuesIterator vectorValuesIterator;
    protected int dimension;
    protected int bytesPerVector;

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
     * Intended to return a vector reference either after deep copy of the vector obtained from {@code getVector}
     * or return the vector itself.
     * <p>
     *   This decision to clone depends on the vector returned based on the type of iterator
     * </p>
     * Running this function can incur latency hence should be absolutely used when necessary.
     * For most of the cases {@link  #getVector()} function should work.
     *
     * @return T an array of byte[], float[] Or a deep copy of it
     * @throws IOException
     */
    public abstract T conditionalCloneVector() throws IOException;

    /**
     * Dimension of vector is returned. Do call getVector function first before calling this function otherwise you will get 0 value.
     * @return int
     */
    public int dimension() {
        assert docId() != -1 && dimension != 0 : "Cannot get dimension before we retrieve a vector from KNNVectorValues";
        return dimension;
    }

    /**
     * Size of a vector in bytes is returned. Do call getVector function first before calling this function otherwise you will get 0 value.
     * @return int
     */
    public int bytesPerVector() {
        assert docId() != -1 && bytesPerVector != 0 : "Cannot get bytesPerVector before we retrieve a vector from KNNVectorValues";
        return bytesPerVector;
    }

    /**
     * Returns the total live docs for KNNVectorValues. This function is broken and doesn't always give the accurate
     * live docs count when iterators are {@link FloatVectorValues}, {@link ByteVectorValues}. Avoid using this iterator,
     * rather use a simple function like this:
     * <pre class="prettyprint">
     *     int liveDocs = 0;
     *     while(vectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
     *         liveDocs++;
     *     }
     * </pre>
     * @return long
     */
    @Deprecated
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
