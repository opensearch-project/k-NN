/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.knn.index.VectorDataType;

import java.util.Map;

/**
 * Mapper used when you dont want to build an underlying KNN struct - you just want to
 * store vectors as doc values
 */
public class FlatVectorFieldMapper extends KNNVectorFieldMapper {

    private final PerDimensionValidator perDimensionValidator;

    public static FlatVectorFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        VectorDataType vectorDataType,
        Integer dimension,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexCreatedVersion
    ) {
        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(fullname, metaValue, vectorDataType, () -> dimension);
        return new FlatVectorFieldMapper(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            indexCreatedVersion
        );
    }

    private FlatVectorFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexCreatedVersion
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo, ignoreMalformed, stored, hasDocValues, indexCreatedVersion, null);
        this.perDimensionValidator = selectPerDimensionValidator(vectorDataType);
        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        this.fieldType.freeze();
    }

    private PerDimensionValidator selectPerDimensionValidator(VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
        }

        return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
    }

    @Override
    protected VectorValidator getVectorValidator() {
        return VectorValidator.NOOP_VECTOR_VALIDATOR;
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        return perDimensionValidator;
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        return PerDimensionProcessor.NOOP_PROCESSOR;
    }
}
