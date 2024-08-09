/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

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

    private final AtomicReference<SpaceType> spaceType;
    private final AtomicReference<MethodComponentContext> methodComponentContext;
    private final AtomicInteger dimension;
    private final AtomicReference<VectorDataType> vectorDataType;

    private final AtomicReference<PerDimensionProcessor> perDimensionProcessor;
    private final AtomicReference<PerDimensionValidator> perDimensionValidator;
    private final AtomicReference<VectorValidator> vectorValidator;

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
                ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
                if (!ModelUtil.isModelCreated(modelMetadata)) {
                    throw new IllegalStateException(String.format("Model ID '%s' is not created.", modelId));
                }
                return modelMetadata.getDimension();
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

        this.spaceType = new AtomicReference<>(null);
        this.methodComponentContext = new AtomicReference<>(null);
        this.dimension = new AtomicInteger(UNSET_MODEL_DIMENSION_IDENTIFIER);
        this.vectorDataType = new AtomicReference<>(null);
        this.perDimensionProcessor = new AtomicReference<>(null);
        this.perDimensionValidator = new AtomicReference<>(null);
        this.vectorValidator = new AtomicReference<>(null);

        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        this.fieldType.putAttribute(MODEL_ID, modelId);
        this.fieldType.freeze();
    }

    @Override
    protected void validatePreparse() {
        super.validatePreparse();
        // For the model field mapper, we cannot validate the model during index creation due to
        // an issue with reading cluster state during mapper creation. So, we need to validate the
        // model when ingestion starts. We do this as lazily as we can
        ModelMetadata modelMetadata = this.modelDao.getMetadata(modelId);

        if (!ModelUtil.isModelCreated(modelMetadata)) {
            throw new IllegalStateException(
                String.format(
                    "Model \"%s\" from %s's mapping is not created. Because the \"%s\" parameter is not updatable, this index will need to be recreated with a valid model.",
                    modelId,
                    simpleName(),
                    MODEL_ID
                )
            );
        }

        maybeInitLazyVariables(modelMetadata);
    }

    private void maybeInitLazyVariables(ModelMetadata modelMetadata) {
        vectorDataType.compareAndExchange(null, modelMetadata.getVectorDataType());
        if (spaceType.get() == null) {
            spaceType.compareAndExchange(null, modelMetadata.getSpaceType());
            spaceType.get().validateVectorDataType(vectorDataType.get());
        }
        methodComponentContext.compareAndExchange(null, modelMetadata.getMethodComponentContext());
        dimension.compareAndExchange(UNSET_MODEL_DIMENSION_IDENTIFIER, modelMetadata.getDimension());
        maybeInitValidatorsAndProcessors();
    }

    private void maybeInitValidatorsAndProcessors() {
        this.vectorValidator.compareAndExchange(null, new SpaceVectorValidator(spaceType.get()));

        if (VectorDataType.BINARY == vectorDataType.get()) {
            this.perDimensionValidator.compareAndExchange(null, PerDimensionValidator.DEFAULT_BIT_VALIDATOR);
            this.perDimensionProcessor.compareAndExchange(null, PerDimensionProcessor.NOOP_PROCESSOR);
            return;
        }

        if (VectorDataType.BYTE == vectorDataType.get()) {
            this.perDimensionValidator.compareAndExchange(null, PerDimensionValidator.DEFAULT_BYTE_VALIDATOR);
            this.perDimensionProcessor.compareAndExchange(null, PerDimensionProcessor.NOOP_PROCESSOR);
            return;
        }

        if (!isFaissSQfp16(methodComponentContext.get())) {
            // Normal float and byte processor
            this.perDimensionValidator.compareAndExchange(null, PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR);
            this.perDimensionProcessor.compareAndExchange(null, PerDimensionProcessor.NOOP_PROCESSOR);
            return;
        }

        this.perDimensionValidator.compareAndExchange(null, PerDimensionValidator.DEFAULT_FP16_VALIDATOR);
        if (!isFaissSQClipToFP16RangeEnabled(
            (MethodComponentContext) methodComponentContext.get().getParameters().get(METHOD_ENCODER_PARAMETER)
        )) {
            this.perDimensionProcessor.compareAndExchange(null, PerDimensionProcessor.NOOP_PROCESSOR);
            return;
        }
        this.perDimensionProcessor.compareAndExchange(null, PerDimensionProcessor.CLIP_TO_FP16_PROCESSOR);
    }

    @Override
    protected VectorValidator getVectorValidator() {
        return vectorValidator.get();
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        return perDimensionValidator.get();
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        return perDimensionProcessor.get();
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        validatePreparse();
        parseCreateField(context, dimension.get(), vectorDataType.get());
    }
}
