/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;

import java.io.IOException;

/**
 * Provides different strategies to extract the vectors from different {@link KNNVectorValuesIterator}
 */
public interface VectorValueExtractorStrategy {

    /**
     * Extract a float vector from KNNVectorValuesIterator.
     * @param iterator {@link KNNVectorValuesIterator}
     * @return float[]
     * @throws IOException exception while retrieving the vectors
     */
    static float[] extractFloatVector(final KNNVectorValuesIterator iterator) throws IOException {
        return iterator.getVectorExtractorStrategy().extract(VectorDataType.FLOAT, iterator);
    }

    /**
     * Extract a byte vector from KNNVectorValuesIterator.
     * @param iterator {@link KNNVectorValuesIterator}
     * @return byte[]
     * @throws IOException exception while retrieving the vectors
     */
    static byte[] extractByteVector(final KNNVectorValuesIterator iterator) throws IOException {
        return iterator.getVectorExtractorStrategy().extract(VectorDataType.BYTE, iterator);
    }

    /**
     * Extract a binary vector which is represented as byte[] from KNNVectorValuesIterator.
     * @param iterator {@link KNNVectorValuesIterator}
     * @return byte[]
     * @throws IOException exception while retrieving the vectors
     */
    static byte[] extractBinaryVector(final KNNVectorValuesIterator iterator) throws IOException {
        return iterator.getVectorExtractorStrategy().extract(VectorDataType.BINARY, iterator);
    }

    /**
     * Extract Vector based on the vector datatype and vector values iterator.
     * @param vectorDataType {@link VectorDataType}
     * @param vectorValuesIterator {@link KNNVectorValuesIterator}
     * @return vector
     * @param <T> could be of type float[], byte[]
     * @throws IOException exception during extracting the vectors
     */
    <T> T extract(VectorDataType vectorDataType, KNNVectorValuesIterator vectorValuesIterator) throws IOException;

    /**
     * Strategy to extract the vector from {@link KNNVectorValuesIterator.DocIdsIteratorValues}
     */
    class DISIVectorExtractor implements VectorValueExtractorStrategy {
        @Override
        public <T> T extract(final VectorDataType vectorDataType, final KNNVectorValuesIterator vectorValuesIterator) throws IOException {
            final DocIdSetIterator docIdSetIterator = vectorValuesIterator.getDocIdSetIterator();

            if (docIdSetIterator instanceof BinaryDocValues) {
                return extractFromBinaryDocValues(vectorDataType, (BinaryDocValues) docIdSetIterator);
            } else if (docIdSetIterator instanceof KnnVectorValues.DocIndexIterator) {
                return extractFromKnnVectorValues(
                    vectorDataType,
                    (KNNVectorValuesIterator.DocIdsIteratorValues) vectorValuesIterator,
                    (KnnVectorValues.DocIndexIterator) docIdSetIterator
                );
            } else {
                throw new IllegalArgumentException(
                    "VectorValuesIterator is not of a valid type. Valid Types are: BinaryDocValues and KnnVectorValues.DocIndexIterator"
                );
            }
        }

        private <T> T extractFromBinaryDocValues(VectorDataType vectorDataType, BinaryDocValues values) throws IOException {
            BytesRef bytesRef = values.binaryValue();
            if (vectorDataType == VectorDataType.FLOAT) {
                return (T) getFloatVectorFromByteRef(bytesRef);
            } else if (vectorDataType == VectorDataType.BYTE || vectorDataType == VectorDataType.BINARY) {
                return (T) ArrayUtil.copyOfSubArray(bytesRef.bytes, bytesRef.offset, bytesRef.offset + bytesRef.length);
            }
            throw new IllegalArgumentException("Invalid vector data type for BinaryDocValues");
        }

        private <T> T extractFromKnnVectorValues(
            VectorDataType vectorDataType,
            KNNVectorValuesIterator.DocIdsIteratorValues docIdsIteratorValues,
            KnnVectorValues.DocIndexIterator docIdSetIterator
        ) throws IOException {
            int ord = docIdSetIterator.index();
            if (ord == docIdsIteratorValues.getLastOrd()) {
                return (T) docIdsIteratorValues.getLastAccessedVector();
            }
            docIdsIteratorValues.setLastOrd(ord);

            if (vectorDataType.isFloatFamily()) {
                FloatVectorValues knnVectorValues = (FloatVectorValues) docIdsIteratorValues.getKnnVectorValues();
                docIdsIteratorValues.setLastAccessedVector(knnVectorValues.vectorValue(ord));
            } else if (vectorDataType == VectorDataType.BYTE || vectorDataType == VectorDataType.BINARY) {
                ByteVectorValues byteVectorValues = (ByteVectorValues) docIdsIteratorValues.getKnnVectorValues();
                docIdsIteratorValues.setLastAccessedVector(byteVectorValues.vectorValue(ord));
            } else {
                throw new IllegalArgumentException("Invalid vector data type for KnnVectorValues");
            }

            return (T) docIdsIteratorValues.getLastAccessedVector();
        }

        private float[] getFloatVectorFromByteRef(final BytesRef bytesRef) {
            final KNNVectorSerializer vectorSerializer = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE;
            return vectorSerializer.byteToFloatArray(bytesRef);
        }
    }

    /**
     * Strategy to extract the vector from {@link KNNVectorValuesIterator.FieldWriterIteratorValues}
     */
    class FieldWriterIteratorVectorExtractor implements VectorValueExtractorStrategy {

        @SuppressWarnings("unchecked")
        @Override
        public <T> T extract(final VectorDataType vectorDataType, final KNNVectorValuesIterator vectorValuesIterator) throws IOException {
            switch (vectorDataType) {
                case FLOAT:
                case HALF_FLOAT:
                    return (T) ((KNNVectorValuesIterator.FieldWriterIteratorValues<float[]>) vectorValuesIterator).vectorsValue();
                case BYTE:
                case BINARY:
                    return (T) ((KNNVectorValuesIterator.FieldWriterIteratorValues<byte[]>) vectorValuesIterator).vectorsValue();
            }
            throw new IllegalArgumentException(
                "Valid Vector data type not passed to extract vector from FieldWriterIteratorVectorExtractor strategy"
            );
        }
    }

}
