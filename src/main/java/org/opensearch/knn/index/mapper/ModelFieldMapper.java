/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.VectorEncoding;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngineResolver;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.SpaceTypeResolver;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;

/**
 * Field mapper for model in mapping
 */
public class ModelFieldMapper extends KNNVectorFieldMapper {

    // If the dimension has not yet been set because we do not have access to model metadata, it will be -1
    public static final int UNSET_MODEL_DIMENSION_IDENTIFIER = -1;

    private PerDimensionProcessor perDimensionProcessor;
    private PerDimensionValidator perDimensionValidator;
    private VectorValidator vectorValidator;

    public static ModelFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        String modelId,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        ModelDao modelDao,
        Version indexCreatedVersion,
        OriginalParameters originalParameters
    ) {
        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(fullname, metaValue, () -> {
            ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);
            KNNMethodConfigContext knnMethodConfigContext = getKNNMethodConfigContextFromModelMetadata(modelId, modelMetadata);
            KNNVectorFieldType.KNNVectorFieldTypeConfig.KNNVectorFieldTypeConfigBuilder builder =
                KNNVectorFieldType.KNNVectorFieldTypeConfig.builder()
                    .dimension(modelMetadata.getDimension())
                    .knnMethodConfigContext(knnMethodConfigContext)
                    .vectorDataType(modelMetadata.getVectorDataType());
            if (knnMethodConfigContext != null && knnMethodConfigContext.getKnnMethodContext() != null) {
                KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodConfigContext.getKnnEngine()
                    .getKNNLibraryIndexingContext(knnMethodConfigContext);
                builder.knnLibrarySearchContext(knnLibraryIndexingContext.getKNNLibrarySearchContext());
            }
            return builder.build();
        }, modelId);
        return new ModelFieldMapper(
            simpleName,
            mappedFieldType,
            modelId,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            modelDao,
            indexCreatedVersion,
            originalParameters
        );
    }

    private ModelFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        String modelId,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        ModelDao modelDao,
        Version indexCreatedVersion,
        OriginalParameters originalParameters
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
            originalParameters
        );
        this.modelDao = modelDao;

        // For the model field mapper, we cannot validate the model during index creation due to
        // an issue with reading cluster state during mapper creation. So, we need to validate the
        // model when ingestion starts. We do this as lazily as we can
        this.perDimensionProcessor = null;
        this.perDimensionValidator = null;
        this.vectorValidator = null;

        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        this.fieldType.putAttribute(MODEL_ID, modelId);
        this.useLuceneBasedVectorField = KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(this.indexCreatedVersion);
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
        // Need to handle BWC case
        if (fieldType().getKnnMethodConfigContext().isEmpty()
            || fieldType().getKnnMethodConfigContext().get().getKnnMethodContext() == null) {
            vectorValidator = new SpaceVectorValidator(fieldType().getKnnMethodConfigContext().get().getSpaceType());
            return;
        }
        KNNMethodConfigContext knnMethodConfigContext = fieldType().getKnnMethodConfigContext().get();
        KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodConfigContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodConfigContext);
        vectorValidator = knnLibraryIndexingContext.getVectorValidator();
    }

    private void initPerDimensionValidator() {
        if (perDimensionValidator != null) {
            return;
        }

        if (fieldType().getKnnMethodConfigContext().isEmpty()) {
            perDimensionValidator = PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
            return;
        }
        KNNMethodConfigContext knnMethodConfigContext = fieldType().getKnnMethodConfigContext().get();

        if (knnMethodConfigContext.getKnnMethodContext() == null) {
            if (knnMethodConfigContext.getVectorDataType() == VectorDataType.BINARY) {
                perDimensionValidator = PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
            } else if (knnMethodConfigContext.getVectorDataType() == VectorDataType.BYTE) {
                perDimensionValidator = PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
            } else {
                perDimensionValidator = PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
            }
            return;
        }

        KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodConfigContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodConfigContext);
        perDimensionValidator = knnLibraryIndexingContext.getPerDimensionValidator();
    }

    private void initPerDimensionProcessor() {
        if (perDimensionProcessor != null) {
            return;
        }

        // Need to handle BWC case
        if (fieldType().getKnnMethodConfigContext().isEmpty()
            || fieldType().getKnnMethodConfigContext().get().getKnnMethodContext() == null) {
            perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }
        KNNMethodConfigContext knnMethodConfigContext = fieldType().getKnnMethodConfigContext().get();
        KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodConfigContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodConfigContext);
        perDimensionProcessor = knnLibraryIndexingContext.getPerDimensionProcessor();
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        validatePreparse();
        KNNMethodConfigContext knnMethodConfigContext = fieldType().getKnnMethodConfigContext().orElse(null);

        if (useLuceneBasedVectorField && knnMethodConfigContext != null) {
            int adjustedDimension = fieldType().getVectorDataType() == VectorDataType.BINARY
                ? fieldType().getDimension() / Byte.SIZE
                : fieldType().getDimension();
            final VectorEncoding encoding = fieldType().getVectorDataType() == VectorDataType.FLOAT
                ? VectorEncoding.FLOAT32
                : VectorEncoding.BYTE;
            fieldType.setVectorAttributes(
                adjustedDimension,
                encoding,
                knnMethodConfigContext.getSpaceType().getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
            );
        } else {
            fieldType.setDocValuesType(DocValuesType.BINARY);
        }

        // Conditionally add quantization config
        if (knnMethodConfigContext != null && knnMethodConfigContext.getKnnMethodContext() != null) {
            KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodConfigContext.getKnnEngine()
                .getKNNLibraryIndexingContext(knnMethodConfigContext);
            QuantizationConfig quantizationConfig = knnLibraryIndexingContext.getQuantizationConfig();
            if (quantizationConfig != null && quantizationConfig != QuantizationConfig.EMPTY) {
                this.fieldType.putAttribute(QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(quantizationConfig));
            }
        }

        parseCreateField(context, fieldType().getDimension(), fieldType().getVectorDataType());
    }

    private static KNNMethodConfigContext getKNNMethodConfigContextFromModelMetadata(String modelId, ModelMetadata modelMetadata) {
        MethodComponentContext methodComponentContext = modelMetadata.getMethodComponentContext();
        if (methodComponentContext == MethodComponentContext.EMPTY) {
            return null;
        }
        // TODO: Need to fix this version check by serializing the model
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(modelMetadata.getVectorDataType())
            .dimension(modelMetadata.getDimension())
            .versionCreated(Version.V_2_14_0)
            .knnMethodContext(ModelUtil.getMethodContextForModel(modelMetadata))
            .modelId(modelId)
            .workloadModeConfig(modelMetadata.getWorkloadModeConfig())
            .compressionConfig(modelMetadata.getCompressionConfig())
            .build();
        knnMethodConfigContext.setSpaceType(SpaceTypeResolver.resolveSpaceType(knnMethodConfigContext));
        knnMethodConfigContext.setKnnEngine(KNNEngineResolver.resolveKNNEngine(knnMethodConfigContext));
        return knnMethodConfigContext;
    }

    private static ModelMetadata getModelMetadata(ModelDao modelDao, String modelId) {
        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (!ModelUtil.isModelCreated(modelMetadata)) {
            throw new IllegalStateException(String.format("Model ID '%s' is not created.", modelId));
        }
        return modelMetadata;
    }
}
