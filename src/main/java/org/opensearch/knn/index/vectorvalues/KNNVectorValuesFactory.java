/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.index.vectorvalues.KNNMergeVectorValues.mergeByteVectorValues;
import static org.opensearch.knn.index.vectorvalues.KNNMergeVectorValues.mergeFloatVectorValues;

/**
 * A factory class that provides various methods to create the {@link KNNVectorValues}.
 */
@Log4j2
public final class KNNVectorValuesFactory {

    /**
     * Returns a {@link KNNVectorValues} for the given {@link DocIdSetIterator} and {@link VectorDataType}
     *
     * @param vectorDataType {@link VectorDataType}
     * @param docIdSetIterator {@link DocIdSetIterator}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(final VectorDataType vectorDataType, final DocIdSetIterator docIdSetIterator) {
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.DocIdsIteratorValues(docIdSetIterator));
    }

    /**
     * Returns a {@link KNNVectorValues} for the given {@link DocIdSetIterator} and a Map of docId and vectors.
     *
     * @param vectorDataType {@link VectorDataType}
     * @param docIdWithFieldSet {@link DocsWithFieldSet}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(
        final VectorDataType vectorDataType,
        final DocsWithFieldSet docIdWithFieldSet,
        final Map<Integer, T> vectors
    ) {
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.FieldWriterIteratorValues<>(docIdWithFieldSet, vectors));
    }

    public static <T> KNNVectorValues<T> getVectorValues(
        final VectorDataType vectorDataType,
        final FieldInfo fieldInfo,
        final MergeState mergeState
    ) {
        try {
            switch (fieldInfo.getVectorEncoding()) {
                case FLOAT32:
                    return getVectorValues(vectorDataType, mergeFloatVectorValues(fieldInfo, mergeState));
                case BYTE:
                    return getVectorValues(vectorDataType, mergeByteVectorValues(fieldInfo, mergeState));
                default:
                    throw new IllegalStateException("Unsupported vector encoding [" + fieldInfo.getVectorEncoding() + "]");
            }
        } catch (final IOException e) {
            log.error("Unable to merge vectors for field [{}]", fieldInfo.getName(), e);
            throw new IllegalStateException("Unable to merge vectors for field [" + fieldInfo.getName() + "]", e);
        }
    }

    /**
     * Returns a {@link KNNVectorValues} for the given {@link FieldInfo} and {@link LeafReader}
     *
     * @param fieldInfo {@link FieldInfo}
     * @param leafReader {@link LeafReader}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(final FieldInfo fieldInfo, final LeafReader leafReader) throws IOException {
        final DocIdSetIterator docIdSetIterator;
        if (fieldInfo.hasVectorValues()) {
            if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
                docIdSetIterator = leafReader.getByteVectorValues(fieldInfo.getName());
            } else if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
                docIdSetIterator = leafReader.getFloatVectorValues(fieldInfo.getName());
            } else {
                throw new IllegalArgumentException("Invalid Vector encoding provided, hence cannot return VectorValues");
            }
        } else {
            docIdSetIterator = DocValues.getBinary(leafReader, fieldInfo.getName());
        }
        final KNNVectorValuesIterator vectorValuesIterator = new KNNVectorValuesIterator.DocIdsIteratorValues(docIdSetIterator);
        return getVectorValues(FieldInfoExtractor.extractVectorDataType(fieldInfo), vectorValuesIterator);
    }

    @VisibleForTesting
    @SuppressWarnings("unchecked")
    static <T> KNNVectorValues<T> getVectorValues(
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
