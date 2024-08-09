/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.isFaissSQClipToFP16RangeEnabled;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.isFaissSQfp16;

/**
 * Field mapper for model in mapping
 */
public class ModelFieldMapper extends KNNVectorFieldMapper {

    // If the dimension has not yet been set because we do not have access to model metadata, it will be -1
    public static final int UNSET_MODEL_DIMENSION_IDENTIFIER = -1;

    private PerDimensionProcessor perDimensionProcessor;
    private PerDimensionValidator perDimensionValidator;
    private VectorValidator vectorValidator;

    private final String modelId;

    public static ModelFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        VectorDataType vectorDataType,
        String modelId,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        ModelDao modelDao,
        Version indexCreatedVersion
    ) {

        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(fullname, metaValue, vectorDataType, new KNNMappingConfig() {
            @Override
            public Optional<String> getModelId() {
                return Optional.of(modelId);
            }

            @Override
            public int getDimension() {
                return getModelMetadata(modelDao, modelId).getDimension();
            }
        });
        return new ModelFieldMapper(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            modelDao,
            indexCreatedVersion
        );
    }

    private ModelFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        ModelDao modelDao,
        Version indexCreatedVersion
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo, ignoreMalformed, stored, hasDocValues, indexCreatedVersion, null);
        KNNMappingConfig annConfig = mappedFieldType.getKnnMappingConfig();
        modelId = annConfig.getModelId().orElseThrow(() -> new IllegalArgumentException("KNN method context cannot be empty"));
        this.modelDao = modelDao;

        // For the model field mapper, we cannot validate the model during index creation due to
        // an issue with reading cluster state during mapper creation. So, we need to validate the
        // model when ingestion starts. We do this as lazily as we can
        this.perDimensionProcessor = null;
        this.perDimensionValidator = null;
        this.vectorValidator = null;

        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        this.fieldType.putAttribute(MODEL_ID, modelId);
        this.fieldType.freeze();
    }

    @Override
    protected VectorValidator getVectorValidator() {
        initVectorValidator();
        return vectorValidator;
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        initPerDimensionValidator();
        return perDimensionValidator;
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        initPerDimensionProcessor();
        return perDimensionProcessor;
    }

    private void initVectorValidator() {
        if (vectorValidator != null) {
            return;
        }
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);
        vectorValidator = new SpaceVectorValidator(modelMetadata.getSpaceType());
    }

    private void initPerDimensionValidator() {
        if (perDimensionValidator != null) {
            return;
        }
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);
        MethodComponentContext methodComponentContext = modelMetadata.getMethodComponentContext();
        VectorDataType dataType = modelMetadata.getVectorDataType();

        if (VectorDataType.BINARY == dataType) {
            perDimensionValidator = PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
            return;
        }

        if (VectorDataType.BYTE == dataType) {
            perDimensionValidator = PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
            return;
        }

        if (!isFaissSQfp16(methodComponentContext)) {
            perDimensionValidator = PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
            return;
        }

        perDimensionValidator = PerDimensionValidator.DEFAULT_FP16_VALIDATOR;
    }

    private void initPerDimensionProcessor() {
        if (perDimensionProcessor != null) {
            return;
        }
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);
        MethodComponentContext methodComponentContext = modelMetadata.getMethodComponentContext();
        VectorDataType dataType = modelMetadata.getVectorDataType();

        if (VectorDataType.BINARY == dataType) {
            perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }

        if (VectorDataType.BYTE == dataType) {
            perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }

        if (!isFaissSQfp16(methodComponentContext)) {
            perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }

        if (!isFaissSQClipToFP16RangeEnabled(
            (MethodComponentContext) methodComponentContext.getParameters().get(METHOD_ENCODER_PARAMETER)
        )) {
            perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }
        perDimensionProcessor = PerDimensionProcessor.CLIP_TO_FP16_PROCESSOR;
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        validatePreparse();
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);
        parseCreateField(context, modelMetadata.getDimension(), modelMetadata.getVectorDataType());
    }

    private static ModelMetadata getModelMetadata(ModelDao modelDao, String modelId) {
        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (!ModelUtil.isModelCreated(modelMetadata)) {
            throw new IllegalStateException(String.format("Model ID '%s' is not created.", modelId));
        }
        return modelMetadata;
    }
}
