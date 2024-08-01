/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.VectorDataType;

import java.util.Map;

/**
 * A factory class that provides various methods to create the {@link KNNVectorValues}.
 */
public final class KNNVectorValuesFactory {

    /**
     * Returns a {@link KNNVectorValues} for the given {@link DocIdSetIterator} and {@link VectorDataType}
     *
     * @param vectorDataType {@link VectorDataType}
     * @param docIdSetIterator {@link DocIdSetIterator}
     * @return {@link KNNVectorValues} of type float[]
     */
    public static <T> KNNVectorValues<T> getVectorValues(final VectorDataType vectorDataType, final DocIdSetIterator docIdSetIterator) {
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.DocIdsIteratorValues(docIdSetIterator));
    }

    /**
     * Returns a {@link KNNVectorValues} for the given {@link DocIdSetIterator} and a Map of docId and vectors.
     *
     * @param vectorDataType {@link VectorDataType}
     * @param docIdWithFieldSet {@link DocsWithFieldSet}
     * @return {@link KNNVectorValues} of type float[]
     */
    public static <T> KNNVectorValues<T> getVectorValues(
        final VectorDataType vectorDataType,
        final DocsWithFieldSet docIdWithFieldSet,
        final Map<Integer, T> vectors
    ) {
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.FieldWriterIteratorValues<T>(docIdWithFieldSet, vectors));
    }

    @SuppressWarnings("unchecked")
    private static <T> KNNVectorValues<T> getVectorValues(
        final VectorDataType vectorDataType,
        final KNNVectorValuesIterator knnVectorValuesIterator
    ) {
        switch (vectorDataType) {
            case FLOAT:
                return (KNNVectorValues<T>) new KNNFloatVectorValues(knnVectorValuesIterator);
            case BYTE:
                return (KNNVectorValues<T>) new KNNByteVectorValues(knnVectorValuesIterator);
            case BINARY:
                return (KNNVectorValues<T>) new KNNBinaryVectorValues(knnVectorValuesIterator);
        }
        throw new IllegalArgumentException("Invalid Vector data type provided, hence cannot return VectorValues");
    }
}
