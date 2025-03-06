/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY;
import static org.opensearch.knn.common.KNNConstants.DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE;

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
        KNNMethodConfigContext knnMethodConfigContext,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        OriginalMappingParameters originalMappingParameters,
        boolean isDerivedSourceEnabled
    ) {
        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
            fullname,
            metaValue,
            knnMethodConfigContext.getVectorDataType(),
            knnMethodConfigContext::getDimension
        );
        return new FlatVectorFieldMapper(
            fullname,
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodConfigContext.getVersionCreated(),
            originalMappingParameters,
            isDerivedSourceEnabled
        );
    }

    private FlatVectorFieldMapper(
        String fullName,
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexCreatedVersion,
        OriginalMappingParameters originalMappingParameters,
        boolean isDerivedSourceEnabled
    ) {
        super(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            indexCreatedVersion,
            originalMappingParameters,
            isDerivedSourceEnabled
        );
        // setting it explicitly false here to ensure that when flatmapper is used Lucene based Vector field is not created.
        this.useLuceneBasedVectorField = false;
        this.perDimensionValidator = selectPerDimensionValidator(vectorDataType);
        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        this.fieldType.setDocValuesType(DocValuesType.BINARY);
        if (KNNVectorFieldMapperUtil.isDeriveSourceForFieldEnabled(isDerivedSourceEnabled, fullName)) {
            this.fieldType.putAttribute(DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY, DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE);
        }
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
