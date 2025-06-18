/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.extern.log4j.Log4j2;
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
@Log4j2
public class ModelFieldMapper extends KNNVectorFieldMapper {

    // If the dimension has not yet been set because we do not have access to model metadata, it will be -1
    public static final int UNSET_MODEL_DIMENSION_IDENTIFIER = -1;

    private PerDimensionProcessor perDimensionProcessor;
    private PerDimensionValidator perDimensionValidator;
    private VectorValidator vectorValidator;
    private VectorTransformer vectorTransformer;

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
        OriginalMappingParameters originalMappingParameters,
        KNNMethodConfigContext knnMethodConfigContext
    ) {

        final KNNMethodContext knnMethodContext = originalMappingParameters.getKnnMethodContext();
        final QuantizationConfig quantizationConfig = knnMethodContext == null
            ? QuantizationConfig.EMPTY
            : knnMethodContext.getKnnEngine()
                .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
                .getQuantizationConfig();

        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(fullname, metaValue, vectorDataType, new KNNMappingConfig() {
            private Integer dimension = null;
            private Mode mode = null;
            private CompressionLevel compressionLevel = null;

            @Override
            public Optional<String> getModelId() {
                return Optional.of(originalMappingParameters.getModelId());
            }

            @Override
            public int getDimension() {
                if (dimension == null) {
                    initFromModelMetadata();
                }

                return dimension;
            }

            @Override
            public Mode getMode() {
                if (mode == null) {
                    initFromModelMetadata();
                }
                return mode;
            }

            @Override
            public CompressionLevel getCompressionLevel() {
                if (compressionLevel == null) {
                    initFromModelMetadata();
                }
                return compressionLevel;
            }

            @Override
            public QuantizationConfig getQuantizationConfig() {
                return quantizationConfig;
            }

            @Override
            public Version getIndexCreatedVersion() {
                return indexCreatedVersion;
            }

            // ModelMetadata relies on cluster state which may not be available during field mapper creation. Thus,
            // we lazily initialize it.
            private void initFromModelMetadata() {
                ModelMetadata modelMetadata = getModelMetadata(modelDao, originalMappingParameters.getModelId());
                dimension = modelMetadata.getDimension();
                mode = modelMetadata.getMode();
                compressionLevel = modelMetadata.getCompressionLevel();
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

    @Override
    protected VectorTransformer getVectorTransformer() {
        // we don't want to call model metadata to get space type and engine for every vector,
        // since getVectorTransformer() will be called once per vector. Hence,
        // we initialize it once, and use it every other time
        initVectorTransformer();
        return this.vectorTransformer;
    }

    /**
     * Initializes the vector transformer for the model field if not already initialized.
     * This method handles the vector transformation configuration based on the model metadata
     * and KNN method context.
     * @throws IllegalStateException if model metadata cannot be retrieved
     */

    private void initVectorTransformer() {
        log.info("in training init vector transformer");
        if (vectorTransformer != null) {
            return;
        }
        log.info("after return statment");
        ModelMetadata modelMetadata = getModelMetadata(modelDao, modelId);

        KNNMethodContext knnMethodContext = getKNNMethodContextFromModelMetadata(modelMetadata);
        KNNMethodConfigContext knnMethodConfigContext = getKNNMethodConfigContextFromModelMetadata(modelMetadata);
        // Need to handle BWC case where method context is not available
        if (knnMethodContext == null || knnMethodConfigContext == null) {
            vectorTransformer = VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER;
            return;
        }
        // get vector transformer from Indexing Context. We want Engine/Library to provide necessary
        // input rather than creating Transformer from the engine and space type. This design
        // decision is taken to make sure that Engine will drive the implementation than Field Mapper.
        KNNLibraryIndexingContext knnLibraryIndexingContext = knnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
        vectorTransformer = knnLibraryIndexingContext.getVectorTransformer();
        log.info("vector Transformer from train: {}", vectorTransformer);
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
            .versionCreated(modelMetadata.getModelVersion())
            .mode(modelMetadata.getMode())
            .compressionLevel(modelMetadata.getCompressionLevel())
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
