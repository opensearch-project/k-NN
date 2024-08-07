/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.getMethodComponentContext;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.isFaissSQClipToFP16RangeEnabled;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.isFaissSQfp16;

/**
 * Field mapper for method definition in mapping
 */
public class MethodFieldMapper extends KNNVectorFieldMapper {

    private PerDimensionProcessor perDimensionProcessor;
    private PerDimensionValidator perDimensionValidator;
    private VectorValidator vectorValidator;

    public static MethodFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        VectorDataType vectorDataType,
        Integer dimension,
        KNNMethodContext knnMethodContext,
        KNNMethodContext originalKNNMethodContext,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexCreatedVersion
    ) {
        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(fullname, metaValue, vectorDataType, new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(knnMethodContext);
            }

            @Override
            public Optional<Integer> getDimension() {
                return Optional.of(dimension);
            }
        });
        return new MethodFieldMapper(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            indexCreatedVersion,
            originalKNNMethodContext
        );
    }

    private MethodFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexVerision,
        KNNMethodContext originalKNNMethodContext
    ) {

        super(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            indexVerision,
            originalKNNMethodContext
        );
        KNNMappingConfig annConfig = mappedFieldType.getKnnMappingConfig();
        KNNMethodContext knnMethodContext = annConfig.getKnnMethodContext()
            .orElseThrow(() -> new IllegalArgumentException("KNN method context cannot be empty"));
        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);

        this.fieldType.putAttribute(
            DIMENSION,
            String.valueOf(annConfig.getDimension().orElseThrow(() -> new IllegalArgumentException("Dimension cannot be empty")))
        );
        this.fieldType.putAttribute(SPACE_TYPE, knnMethodContext.getSpaceType().getValue());
        this.fieldType.putAttribute(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());

        KNNEngine knnEngine = knnMethodContext.getKnnEngine();
        this.fieldType.putAttribute(KNN_ENGINE, knnEngine.getName());

        try {
            Map<String, Object> libParams = knnEngine.getKNNLibraryIndexingContext(knnMethodContext).getLibraryParameters();
            this.fieldType.putAttribute(PARAMETERS, XContentFactory.jsonBuilder().map(libParams).toString());
        } catch (IOException ioe) {
            throw new RuntimeException(String.format("Unable to create KNNVectorFieldMapper: %s", ioe));
        }

        this.fieldType.freeze();
        initValidatorsAndProcessors(knnMethodContext);
        knnMethodContext.getSpaceType().validateVectorDataType(vectorDataType);
    }

    private void initValidatorsAndProcessors(KNNMethodContext knnMethodContext) {
        this.vectorValidator = new SpaceVectorValidator(knnMethodContext.getSpaceType());

        if (VectorDataType.BINARY == vectorDataType) {
            this.perDimensionValidator = PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
            this.perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            this.perDimensionValidator = PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
            this.perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }

        MethodComponentContext methodComponentContext = getMethodComponentContext(knnMethodContext);
        if (!isFaissSQfp16(methodComponentContext)) {
            // Normal float and byte processor
            this.perDimensionValidator = PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
            this.perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }

        this.perDimensionValidator = PerDimensionValidator.DEFAULT_FP16_VALIDATOR;

        if (!isFaissSQClipToFP16RangeEnabled(
            (MethodComponentContext) methodComponentContext.getParameters().get(METHOD_ENCODER_PARAMETER)
        )) {
            this.perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }

        this.perDimensionProcessor = PerDimensionProcessor.CLIP_TO_FP16_PROCESSOR;
    }

    @Override
    protected VectorValidator getVectorValidator() {
        return vectorValidator;
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        return perDimensionValidator;
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        return perDimensionProcessor;
    }
}
