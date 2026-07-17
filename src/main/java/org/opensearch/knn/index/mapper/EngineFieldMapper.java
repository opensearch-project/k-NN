/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 *  Field mapper for all supported engines.
 */
public class EngineFieldMapper extends KNNVectorFieldMapper {

    private final FieldType vectorFieldType;
    private final PerDimensionProcessor perDimensionProcessor;
    private final PerDimensionValidator perDimensionValidator;
    private final VectorValidator vectorValidator;
    private final VectorTransformer vectorTransformer;
    private final EngineFieldStrategy fieldStrategy;

    public static EngineFieldMapper createFieldMapper(
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
        Version indexCreatedVersion
    ) {
        KNNMethodContext methodContext = originalMappingParameters.getResolvedKnnMethodContext();
        KNNLibraryIndexingContext libraryContext = methodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(methodContext, knnMethodConfigContext);
        ResolvedIndexSpec resolvedSpec = libraryContext.getResolvedSpec();

        KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
            fullname,
            metaValue,
            knnMethodConfigContext.getVectorDataType(),
            new KNNMappingConfig() {
                @Override
                public Optional<KNNMethodContext> getKnnMethodContext() {
                    return Optional.of(methodContext);
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
                public Version getIndexCreatedVersion() {
                    return knnMethodConfigContext.getVersionCreated();
                }

                @Override
                public QuantizationConfig getQuantizationConfig() {
                    return Optional.ofNullable(libraryContext)
                        .map(KNNLibraryIndexingContext::getQuantizationConfig)
                        .orElse(QuantizationConfig.EMPTY);
                }

                @Override
                public KNNLibraryIndexingContext getKnnLibraryIndexingContext() {
                    return libraryContext;
                }
            },
            indexCreatedVersion,
            resolvedSpec
        );

        return new EngineFieldMapper(
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

    private EngineFieldMapper(
        String name,
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
            name,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodConfigContext.getVersionCreated(),
            originalMappingParameters
        );
        updateEngineStats();

        KNNMappingConfig knnMappingConfig = mappedFieldType.getKnnMappingConfig();
        VectorDataType vectorDataType = mappedFieldType.getVectorDataType();
        KNNMethodContext resolvedKnnMethodContext = originalMappingParameters.getResolvedKnnMethodContext();
        KNNEngine knnEngine = resolvedKnnMethodContext.getKnnEngine();
        KNNLibraryIndexingContext knnLibraryIndexingContext = knnEngine.getKNNLibraryIndexingContext(
            resolvedKnnMethodContext,
            knnMethodConfigContext
        );

        this.fieldStrategy = knnEngine.getFieldStrategy();
        FieldTypeConfig config = fieldStrategy.buildFieldTypeConfig(
            knnMappingConfig,
            resolvedKnnMethodContext,
            knnLibraryIndexingContext,
            vectorDataType,
            indexCreatedVersion,
            hasDocValues
        );

        this.fieldType = config.getFieldType();
        this.vectorFieldType = config.getVectorFieldType();
        this.vectorTransformer = config.getVectorTransformer();
        this.useLuceneBasedVectorField = config.isUseLuceneBasedVectorField();

        this.perDimensionProcessor = knnLibraryIndexingContext.getPerDimensionProcessor();
        this.perDimensionValidator = knnLibraryIndexingContext.getPerDimensionValidator();
        this.vectorValidator = knnLibraryIndexingContext.getVectorValidator();
    }

    @Override
    protected List<Field> getFieldsForFloatVector(final float[] array, boolean isDerivedSourceEnabled) {
        List<Field> fields = fieldStrategy.createFloatFields(
            name(),
            array,
            fieldType,
            vectorFieldType,
            stored,
            hasDocValues,
            isDerivedSourceEnabled
        );
        if (fields != null) {
            return fields;
        }
        return super.getFieldsForFloatVector(array, isDerivedSourceEnabled);
    }

    @Override
    protected List<Field> getFieldsForByteVector(final byte[] array, boolean isDerivedSourceEnabled) {
        List<Field> fields = fieldStrategy.createByteFields(
            name(),
            array,
            fieldType,
            vectorFieldType,
            stored,
            hasDocValues,
            isDerivedSourceEnabled
        );
        if (fields != null) {
            return fields;
        }
        return super.getFieldsForByteVector(array, isDerivedSourceEnabled);
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

    @Override
    protected VectorTransformer getVectorTransformer() {
        if (vectorTransformer == null) {
            return super.getVectorTransformer();
        }
        return vectorTransformer;
    }

    @Override
    void updateEngineStats() {
        Optional.ofNullable(originalMappingParameters)
            .ifPresent(params -> params.getResolvedKnnMethodContext().getKnnEngine().setInitialized(true));
    }
}
