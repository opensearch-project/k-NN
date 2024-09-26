/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.VectorEncoding;
import org.opensearch.common.Explicit;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Field mapper for method definition in mapping
 */
public class MethodFieldMapper extends KNNVectorFieldMapper {

    private final PerDimensionProcessor perDimensionProcessor;
    private final PerDimensionValidator perDimensionValidator;
    private final VectorValidator vectorValidator;

    public static MethodFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        KNNMethodConfigContext knnMethodConfigContext,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        OriginalMappingParameters originalMappingParameters
    ) {

        KNNMethodContext knnMethodContext = originalMappingParameters.getResolvedKnnMethodContext();
        QuantizationConfig quantizationConfig = knnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext)
            .getQuantizationConfig();

        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
            fullname,
            metaValue,
            knnMethodConfigContext.getVectorDataType(),
            new KNNMappingConfig() {
                @Override
                public Optional<KNNMethodContext> getKnnMethodContext() {
                    return Optional.of(originalMappingParameters.getResolvedKnnMethodContext());
                }

                @Override
                public int getDimension() {
                    return knnMethodConfigContext.getDimension();
                }

                @Override
                public Mode getMode() {
                    return Mode.fromName(originalMappingParameters.getMode());
                }

                @Override
                public CompressionLevel getCompressionLevel() {
                    return knnMethodConfigContext.getCompressionLevel();
                }

                @Override
                public QuantizationConfig getQuantizationConfig() {
                    return quantizationConfig;
                }
            }
        );
        return new MethodFieldMapper(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodConfigContext,
            originalMappingParameters
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
        KNNMethodConfigContext knnMethodConfigContext,
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
            knnMethodConfigContext.getVersionCreated(),
            originalMappingParameters
        );
        this.useLuceneBasedVectorField = KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(indexCreatedVersion);
        KNNMappingConfig knnMappingConfig = mappedFieldType.getKnnMappingConfig();
        KNNMethodContext resolvedKnnMethodContext = originalMappingParameters.getResolvedKnnMethodContext();
        KNNEngine knnEngine = resolvedKnnMethodContext.getKnnEngine();
        KNNLibraryIndexingContext knnLibraryIndexingContext = knnEngine.getKNNLibraryIndexingContext(
            resolvedKnnMethodContext,
            knnMethodConfigContext
        );
        QuantizationConfig quantizationConfig = knnLibraryIndexingContext.getQuantizationConfig();

        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        this.fieldType.putAttribute(DIMENSION, String.valueOf(knnMappingConfig.getDimension()));
        this.fieldType.putAttribute(SPACE_TYPE, resolvedKnnMethodContext.getSpaceType().getValue());
        // Conditionally add quantization config
        if (quantizationConfig != null && quantizationConfig != QuantizationConfig.EMPTY) {
            this.fieldType.putAttribute(QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(quantizationConfig));
        }

        this.fieldType.putAttribute(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        this.fieldType.putAttribute(KNN_ENGINE, knnEngine.getName());

        try {
            this.fieldType.putAttribute(
                PARAMETERS,
                XContentFactory.jsonBuilder().map(knnLibraryIndexingContext.getLibraryParameters()).toString()
            );
        } catch (IOException ioe) {
            throw new RuntimeException(String.format("Unable to create KNNVectorFieldMapper: %s", ioe));
        }

        if (useLuceneBasedVectorField) {
            int adjustedDimension = mappedFieldType.vectorDataType == VectorDataType.BINARY
                ? knnMappingConfig.getDimension() / 8
                : knnMappingConfig.getDimension();
            final VectorEncoding encoding = mappedFieldType.vectorDataType == VectorDataType.FLOAT
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

        this.fieldType.freeze();
        this.perDimensionProcessor = knnLibraryIndexingContext.getPerDimensionProcessor();
        this.perDimensionValidator = knnLibraryIndexingContext.getPerDimensionValidator();
        this.vectorValidator = knnLibraryIndexingContext.getVectorValidator();
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
