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

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Enum contains data_type of vectors and right now only supported for lucene engine in k-NN plugin.
 * We have two vector data_types, one is float (default) and the other one is byte.
 */
@AllArgsConstructor
public enum VectorDataType {
    BYTE("byte") {

        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            return KnnByteVectorField.createFieldType(dimension, vectorSimilarityFunction);
        }
    },
    FLOAT("float") {

        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            return KnnVectorField.createFieldType(dimension, vectorSimilarityFunction);
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
}
