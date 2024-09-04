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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

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

    private final String modelId;

    public static ModelFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        VectorDataType vectorDataType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        ModelDao modelDao,
        Version indexCreatedVersion,
        OriginalMappingParameters originalMappingParameters
    ) {

        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(fullname, metaValue, vectorDataType, new KNNMappingConfig() {
            @Override
            public Optional<String> getModelId() {
                return Optional.of(originalMappingParameters.getModelId());
            }

            @Override
            public int getDimension() {
                return getModelMetadata(modelDao, originalMappingParameters.getModelId()).getDimension();
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
            indexCreatedVersion,
            originalMappingParameters
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
        Version indexCreatedVersion,
        OriginalMappingParameters originalMappingParameters
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
            originalMappingParameters
        );
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
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);

        KNNMethodContext knnMethodContext = getKNNMethodContextFromModelMetadata(modelMetadata);
        KNNMethodConfigContext knnMethodConfigContext = getKNNMethodConfigContextFromModelMetadata(modelMetadata);
        // Need to handle BWC case
        if (knnMethodContext == null || knnMethodConfigContext == null) {
            vectorValidator = new SpaceVectorValidator(modelMetadata.getSpaceType());
            return;
        }

        KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
        vectorValidator = knnLibraryIndexingContext.getVectorValidator();
    }

    private void initPerDimensionValidator() {
        if (perDimensionValidator != null) {
            return;
        }
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);

        KNNMethodContext knnMethodContext = getKNNMethodContextFromModelMetadata(modelMetadata);
        KNNMethodConfigContext knnMethodConfigContext = getKNNMethodConfigContextFromModelMetadata(modelMetadata);
        // Need to handle BWC case
        if (knnMethodContext == null || knnMethodConfigContext == null) {
            if (modelMetadata.getVectorDataType() == VectorDataType.BINARY) {
                perDimensionValidator = PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
            } else if (modelMetadata.getVectorDataType() == VectorDataType.BYTE) {
                perDimensionValidator = PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
            } else {
                perDimensionValidator = PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
            }

            return;
        }

        KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
        perDimensionValidator = knnLibraryIndexingContext.getPerDimensionValidator();
    }

    private void initPerDimensionProcessor() {
        if (perDimensionProcessor != null) {
            return;
        }
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);

        KNNMethodContext knnMethodContext = getKNNMethodContextFromModelMetadata(modelMetadata);
        KNNMethodConfigContext knnMethodConfigContext = getKNNMethodConfigContextFromModelMetadata(modelMetadata);
        // Need to handle BWC case
        if (knnMethodContext == null || knnMethodConfigContext == null) {
            perDimensionProcessor = PerDimensionProcessor.NOOP_PROCESSOR;
            return;
        }

        KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
        perDimensionProcessor = knnLibraryIndexingContext.getPerDimensionProcessor();
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        validatePreparse();
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);
        if (useLuceneBasedVectorField) {
            int adjustedDimension = modelMetadata.getVectorDataType() == VectorDataType.BINARY
                ? modelMetadata.getDimension() / Byte.SIZE
                : modelMetadata.getDimension();
            final VectorEncoding encoding = modelMetadata.getVectorDataType() == VectorDataType.FLOAT
                ? VectorEncoding.FLOAT32
                : VectorEncoding.BYTE;
            fieldType.setVectorAttributes(
                adjustedDimension,
                encoding,
                SpaceType.DEFAULT.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
            );
        } else {
            fieldType.setDocValuesType(DocValuesType.BINARY);
        }

        // Conditionally add quantization config
        KNNMethodContext knnMethodContext = getKNNMethodContextFromModelMetadata(modelMetadata);
        KNNMethodConfigContext knnMethodConfigContext = getKNNMethodConfigContextFromModelMetadata(modelMetadata);
        if (knnMethodContext != null && knnMethodConfigContext != null) {
            KNNLibraryIndexingContext knnLibraryIndexingContext = modelMetadata.getKnnEngine()
                .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
            QuantizationConfig quantizationConfig = knnLibraryIndexingContext.getQuantizationConfig();
            if (quantizationConfig != null && quantizationConfig != QuantizationConfig.EMPTY) {
                this.fieldType.putAttribute(QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(quantizationConfig));
            }
        }

        parseCreateField(context, modelMetadata.getDimension(), modelMetadata.getVectorDataType());
    }

    private static KNNMethodContext getKNNMethodContextFromModelMetadata(ModelMetadata modelMetadata) {
        MethodComponentContext methodComponentContext = modelMetadata.getMethodComponentContext();
        if (methodComponentContext == MethodComponentContext.EMPTY) {
            return null;
        }
        return new KNNMethodContext(modelMetadata.getKnnEngine(), modelMetadata.getSpaceType(), methodComponentContext);
    }

    private static KNNMethodConfigContext getKNNMethodConfigContextFromModelMetadata(ModelMetadata modelMetadata) {
        MethodComponentContext methodComponentContext = modelMetadata.getMethodComponentContext();
        if (methodComponentContext == MethodComponentContext.EMPTY) {
            return null;
        }
        // TODO: Need to fix this version check by serializing the model
        return KNNMethodConfigContext.builder()
            .vectorDataType(modelMetadata.getVectorDataType())
            .dimension(modelMetadata.getDimension())
            .versionCreated(Version.V_2_14_0)
            .build();
    }

    private static ModelMetadata getModelMetadata(ModelDao modelDao, String modelId) {
        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (!ModelUtil.isModelCreated(modelMetadata)) {
            throw new IllegalStateException(String.format("Model ID '%s' is not created.", modelId));
        }
        return modelMetadata;
    }
}
