/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE;

/**
 * Enum contains data_type of vectors and right now only supported for lucene engine in k-NN plugin.
 * We have two vector data_types, one is float (default) and the other one is byte.
 */
public enum VectorDataType {
    BYTE("byte") {
        /**
         * @param dimension  Dimension of the vector
         * @param vectorSimilarityFunction VectorSimilarityFunction for a given spaceType
         * @return FieldType of type KnnByteVectorField
         */
        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            return KnnByteVectorField.createFieldType(dimension, vectorSimilarityFunction);
        }

        /**
         * @param knnEngine KNNEngine
         * @return DocValues FieldType of type Binary and with BYTE VectorEncoding
         */
        @Override
        public FieldType buildDocValuesFieldType(KNNEngine knnEngine) {
            IndexableFieldType indexableFieldType = new IndexableFieldType() {
                @Override
                public boolean stored() {
                    return false;
                }

                @Override
                public boolean tokenized() {
                    return true;
                }

                @Override
                public boolean storeTermVectors() {
                    return false;
                }

                @Override
                public boolean storeTermVectorOffsets() {
                    return false;
                }

                @Override
                public boolean storeTermVectorPositions() {
                    return false;
                }

                @Override
                public boolean storeTermVectorPayloads() {
                    return false;
                }

                @Override
                public boolean omitNorms() {
                    return false;
                }

                @Override
                public IndexOptions indexOptions() {
                    return IndexOptions.NONE;
                }

                @Override
                public DocValuesType docValuesType() {
                    return DocValuesType.NONE;
                }

                @Override
                public int pointDimensionCount() {
                    return 0;
                }

                @Override
                public int pointIndexDimensionCount() {
                    return 0;
                }

                @Override
                public int pointNumBytes() {
                    return 0;
                }

                @Override
                public int vectorDimension() {
                    return 0;
                }

                @Override
                public VectorEncoding vectorEncoding() {
                    return VectorEncoding.BYTE;
                }

                @Override
                public VectorSimilarityFunction vectorSimilarityFunction() {
                    return VectorSimilarityFunction.EUCLIDEAN;
                }

                @Override
                public Map<String, String> getAttributes() {
                    return null;
                }
            };
            FieldType field = new FieldType(indexableFieldType);
            field.putAttribute(KNN_ENGINE, knnEngine.getName());
            field.setDocValuesType(DocValuesType.BINARY);
            field.freeze();
            return field;
        }
    },
    FLOAT("float") {
        /**
         * @param dimension  Dimension of the vector
         * @param vectorSimilarityFunction VectorSimilarityFunction for a given spaceType
         * @return  FieldType of type KnnFloatVectorField
         */
        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            return KnnVectorField.createFieldType(dimension, vectorSimilarityFunction);
        }

        /**
         * @param knnEngine  KNNEngine
         * @return  DocValues FieldType of type Binary and with FLOAT32 VectorEncoding
         */
        @Override
        public FieldType buildDocValuesFieldType(KNNEngine knnEngine) {
            FieldType field = new FieldType();
            field.putAttribute(KNN_ENGINE, knnEngine.getName());
            field.setDocValuesType(DocValuesType.BINARY);
            field.freeze();
            return field;
        }

    };

    private final String value;

    VectorDataType(String value) {
        this.value = value;
    }

    /**
     * Get VectorDataType name
     *
     * @return  name
     */
    public String getValue() {
        return value;
    }

    public abstract FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction);

    public abstract FieldType buildDocValuesFieldType(KNNEngine knnEngine);

    /**
     * @return  Set of names of all the supporting VectorDataTypes
     */
    public static Set<String> getValues() {
        Set<String> values = new HashSet<>();

        for (VectorDataType dataType : VectorDataType.values()) {
            values.add(dataType.getValue());
        }
        return values;
    }

    /**
     * Validates if given VectorDataType is in the list of supported data types.
     * @param vectorDataType VectorDataType
     * @return  the same VectorDataType if it is in the supported values else throw exception.
     */
    public static VectorDataType get(String vectorDataType) {
        String supportedTypes = String.join(",", getValues());
        Objects.requireNonNull(
            vectorDataType,
            String.format("[{}] should not be null. Supported types are [{}]", VECTOR_DATA_TYPE, supportedTypes)
        );
        for (VectorDataType currentDataType : VectorDataType.values()) {
            if (currentDataType.getValue().equalsIgnoreCase(vectorDataType)) {
                return currentDataType;
            }
        }
        throw new IllegalArgumentException(
            String.format(
                "[%s] field was set as [%s] in index mapping. But, supported values are [%s]",
                VECTOR_DATA_TYPE,
                vectorDataType,
                supportedTypes
            )
        );
    }

    /**
     * Validate the float vector values if it is a number and in the finite range.
     *
     * @param value  float vector value
     */
    public static void validateFloatVectorValues(float value) {
        if (Float.isNaN(value)) {
            throw new IllegalArgumentException("KNN vector values cannot be NaN");
        }

        if (Float.isInfinite(value)) {
            throw new IllegalArgumentException("KNN vector values cannot be infinity");
        }
    }

    /**
     * Validate the float vector value in the byte range if it is a finite number,
     * with no decimal values and in the byte range of [-128 to 127].
     *
     * @param value  float value in byte range
     */
    public static void validateByteVectorValues(float value) {
        validateFloatVectorValues(value);
        if (value % 1 != 0) {
            throw new IllegalArgumentException(
                "[data_type] field was set as [byte] in index mapping. But, KNN vector values are floats instead of byte integers"
            );
        }
        if ((int) value < Byte.MIN_VALUE || (int) value > Byte.MAX_VALUE) {
            throw new IllegalArgumentException(
                String.format(
                    "[%s] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [{}, {}]",
                    VECTOR_DATA_TYPE,
                    VectorDataType.BYTE.getValue(),
                    Byte.MIN_VALUE,
                    Byte.MAX_VALUE
                )
            );
        }
    }

    /**
     * Validate if the given vector size matches with the dimension provided in mapping.
     *
     * @param dimension dimension of vector
     * @param vectorSize size of the vector
     */
    public static void validateVectorDimension(int dimension, int vectorSize) {
        if (dimension != vectorSize) {
            String errorMessage = String.format("Vector dimension mismatch. Expected: %d, Given: %d", dimension, vectorSize);
            throw new IllegalArgumentException(errorMessage);
        }

    }

    /**
     * Validates and throws exception if data_type field is set in the index mapping
     * using any VectorDataType (other than float, which is default) with any engine (except lucene).
     *
     * @param knnMethodContext KNNMethodContext Parameter
     * @param vectorDataType VectorDataType Parameter
     */
    public static void validateVectorDataType_Engine(
        ParametrizedFieldMapper.Parameter<KNNMethodContext> knnMethodContext,
        ParametrizedFieldMapper.Parameter<VectorDataType> vectorDataType
    ) {
        if (vectorDataType.getValue() != DEFAULT_VECTOR_DATA_TYPE
            && (knnMethodContext.get() == null || knnMethodContext.getValue().getKnnEngine() != KNNEngine.LUCENE)) {
            throw new IllegalArgumentException(String.format("[%s] is only supported for [%s] engine", VECTOR_DATA_TYPE, LUCENE_NAME));
        }
    }
}
