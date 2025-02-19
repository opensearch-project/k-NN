/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.Map;

/**
 * A factory class that provides various methods to create the {@link KNNVectorValues}.
 */
public final class KNNVectorValuesFactory {

    /**
     * Returns a {@link KNNVectorValues} for the given {@link DocIdSetIterator} and {@link VectorDataType}
     *
     * @param vectorDataType {@link VectorDataType}
     * @param knnVectorValues {@link KnnVectorValues}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(final VectorDataType vectorDataType, final KnnVectorValues knnVectorValues) {
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.DocIdsIteratorValues(knnVectorValues));
    }

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
        return getVectorValues(vectorDataType, new KNNVectorValuesIterator.FieldWriterIteratorValues<T>(docIdWithFieldSet, vectors));
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
        if (!fieldInfo.hasVectorValues()) {
            docIdSetIterator = DocValues.getBinary(leafReader, fieldInfo.getName());
            final KNNVectorValuesIterator vectorValuesIterator = new KNNVectorValuesIterator.DocIdsIteratorValues(docIdSetIterator);
            return getVectorValues(FieldInfoExtractor.extractVectorDataType(fieldInfo), vectorValuesIterator);
        }
        if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
            return getVectorValues(
                FieldInfoExtractor.extractVectorDataType(fieldInfo),
                new KNNVectorValuesIterator.DocIdsIteratorValues(leafReader.getByteVectorValues(fieldInfo.getName()))
            );
        } else if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
            return getVectorValues(
                FieldInfoExtractor.extractVectorDataType(fieldInfo),
                new KNNVectorValuesIterator.DocIdsIteratorValues(leafReader.getFloatVectorValues(fieldInfo.getName()))
            );
        } else {
            throw new IllegalArgumentException("Invalid Vector encoding provided, hence cannot return VectorValues");
        }
    }

    /**
     * Returns a {@link KNNVectorValues} for the given {@link FieldInfo} and {@link LeafReader}
     *
     * @param fieldInfo {@link FieldInfo}
     * @param docValuesProducer {@link DocValuesProducer}
     * @param knnVectorsReader {@link KnnVectorsReader}
     * @return {@link KNNVectorValues}
     */
    public static <T> KNNVectorValues<T> getVectorValues(
        final FieldInfo fieldInfo,
        final DocValuesProducer docValuesProducer,
        final KnnVectorsReader knnVectorsReader
    ) throws IOException {
        if (fieldInfo.hasVectorValues() && knnVectorsReader != null) {
            final KnnVectorValues knnVectorValues;
            if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
                knnVectorValues = knnVectorsReader.getByteVectorValues(fieldInfo.getName());
            } else if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
                knnVectorValues = knnVectorsReader.getFloatVectorValues(fieldInfo.getName());
            } else {
                throw new IllegalArgumentException("Invalid Vector encoding provided, hence cannot return VectorValues");
            }
            return getVectorValues(
                FieldInfoExtractor.extractVectorDataType(fieldInfo),
                new KNNVectorValuesIterator.DocIdsIteratorValues(knnVectorValues)
            );
        } else if (docValuesProducer != null) {
            return getVectorValues(
                FieldInfoExtractor.extractVectorDataType(fieldInfo),
                new KNNVectorValuesIterator.DocIdsIteratorValues(docValuesProducer.getBinary(fieldInfo))
            );
        } else {
            throw new IllegalArgumentException("Field does not have vector values and DocValues");
        }
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
