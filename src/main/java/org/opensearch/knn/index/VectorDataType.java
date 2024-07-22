/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Enum contains data_type of vectors
 * Lucene supports byte and float data type
 * NMSLib supports only float data type
 * Faiss supports binary and float data type
 */
@AllArgsConstructor
public enum VectorDataType {
    BINARY("binary") {

        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            throw new IllegalStateException("Unsupported method");
        }

        @Override
        public float[] getVectorFromBytesRef(BytesRef binaryValue) {
            float[] vector = new float[binaryValue.length];
            int i = 0;
            int j = binaryValue.offset;

            while (i < binaryValue.length) {
                vector[i++] = binaryValue.bytes[j++];
            }
            return vector;
        }
    },
    BYTE("byte") {

        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            return KnnByteVectorField.createFieldType(dimension, vectorSimilarityFunction);
        }

        @Override
        public float[] getVectorFromBytesRef(BytesRef binaryValue) {
            float[] vector = new float[binaryValue.length];
            int i = 0;
            int j = binaryValue.offset;

            while (i < binaryValue.length) {
                vector[i++] = binaryValue.bytes[j++];
            }
            return vector;
        }
    },
    FLOAT("float") {

        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            return KnnVectorField.createFieldType(dimension, vectorSimilarityFunction);
        }

        @Override
        public float[] getVectorFromBytesRef(BytesRef binaryValue) {
            final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByBytesRef(binaryValue);
            return vectorSerializer.byteToFloatArray(binaryValue);
        }

    };

    public static final String SUPPORTED_VECTOR_DATA_TYPES = Arrays.stream(VectorDataType.values())
        .map(VectorDataType::getValue)
        .collect(Collectors.joining(","));
    @Getter
    private final String value;

    /**
     * Creates a KnnVectorFieldType based on the VectorDataType using the provided dimension and
     * VectorSimilarityFunction.
     *
     * @param dimension Dimension of the vector
     * @param vectorSimilarityFunction VectorSimilarityFunction for a given spaceType
     * @return FieldType
     */
    public abstract FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction);

    /**
     * Deserializes float vector from BytesRef.
     *
     * @param binaryValue Binary Value
     * @return float vector deserialized from binary value
     */
    public abstract float[] getVectorFromBytesRef(BytesRef binaryValue);

    /**
     * Validates if given VectorDataType is in the list of supported data types.
     * @param vectorDataType VectorDataType
     * @return  the same VectorDataType if it is in the supported values
     * throws Exception if an invalid value is provided.
     */
    public static VectorDataType get(String vectorDataType) {
        Objects.requireNonNull(
            vectorDataType,
            String.format(
                Locale.ROOT,
                "[%s] should not be null. Supported types are [%s]",
                VECTOR_DATA_TYPE_FIELD,
                SUPPORTED_VECTOR_DATA_TYPES
            )
        );
        try {
            return VectorDataType.valueOf(vectorDataType.toUpperCase(Locale.ROOT));
        } catch (Exception e) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Invalid value provided for [%s] field. Supported values are [%s]",
                    VECTOR_DATA_TYPE_FIELD,
                    SUPPORTED_VECTOR_DATA_TYPES
                )
            );
        }
    }

    public static VectorDataType DEFAULT = FLOAT;
}
