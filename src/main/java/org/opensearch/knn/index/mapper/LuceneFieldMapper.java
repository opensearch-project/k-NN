/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.opensearch.common.Explicit;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForByteVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForFloatVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.buildDocValuesFieldType;

/**
 * Field mapper for case when Lucene has been set as an engine.
 */
public class LuceneFieldMapper extends KNNVectorFieldMapper {

    /** FieldType used for initializing VectorField, which is used for creating binary doc values. **/
    private final FieldType vectorFieldType;

    private final PerDimensionProcessor perDimensionProcessor;
    private final PerDimensionValidator perDimensionValidator;
    private final VectorValidator vectorValidator;

    static LuceneFieldMapper createFieldMapper(
        String fullname,
        Map<String, String> metaValue,
        KNNMethodConfigContext knnMethodConfigContext,
        CreateLuceneFieldMapperInput createLuceneFieldMapperInput,
        OriginalMappingParameters originalMappingParameters
    ) {
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
                    return knnMethodConfigContext.getMode();
                }

                @Override
                public CompressionLevel getCompressionLevel() {
                    return knnMethodConfigContext.getCompressionLevel();
                }
            }
        );

        return new LuceneFieldMapper(mappedFieldType, createLuceneFieldMapperInput, knnMethodConfigContext, originalMappingParameters);
    }

    private LuceneFieldMapper(
        final KNNVectorFieldType mappedFieldType,
        final CreateLuceneFieldMapperInput input,
        KNNMethodConfigContext knnMethodConfigContext,
        OriginalMappingParameters originalMappingParameters
    ) {
        super(
            input.getName(),
            mappedFieldType,
            input.getMultiFields(),
            input.getCopyTo(),
            input.getIgnoreMalformed(),
            input.isStored(),
            input.isHasDocValues(),
            knnMethodConfigContext.getVersionCreated(),
            originalMappingParameters
        );
        KNNMappingConfig knnMappingConfig = mappedFieldType.getKnnMappingConfig();
        KNNMethodContext resolvedKnnMethodContext = originalMappingParameters.getResolvedKnnMethodContext();
        VectorDataType vectorDataType = mappedFieldType.getVectorDataType();

        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = resolvedKnnMethodContext.getSpaceType()
            .getKnnVectorSimilarityFunction();

        this.fieldType = vectorDataType.createKnnVectorFieldType(knnMappingConfig.getDimension(), knnVectorSimilarityFunction);

        if (this.hasDocValues) {
            this.vectorFieldType = buildDocValuesFieldType(resolvedKnnMethodContext.getKnnEngine());
        } else {
            this.vectorFieldType = null;
        }

        KNNLibraryIndexingContext knnLibraryIndexingContext = resolvedKnnMethodContext.getKnnEngine()
            .getKNNLibraryIndexingContext(resolvedKnnMethodContext, knnMethodConfigContext);
        this.perDimensionProcessor = knnLibraryIndexingContext.getPerDimensionProcessor();
        this.perDimensionValidator = knnLibraryIndexingContext.getPerDimensionValidator();
        this.vectorValidator = knnLibraryIndexingContext.getVectorValidator();
    }

    @Override
    protected List<Field> getFieldsForFloatVector(final float[] array) {
        final List<Field> fieldsToBeAdded = new ArrayList<>();
        fieldsToBeAdded.add(new KnnFloatVectorField(name(), array, fieldType));

        if (hasDocValues && vectorFieldType != null) {
            fieldsToBeAdded.add(new VectorField(name(), array, vectorFieldType));
        }

        if (this.stored) {
            fieldsToBeAdded.add(createStoredFieldForFloatVector(name(), array));
        }
        return fieldsToBeAdded;
    }

    @Override
    protected List<Field> getFieldsForByteVector(final byte[] array) {
        final List<Field> fieldsToBeAdded = new ArrayList<>();
        fieldsToBeAdded.add(new KnnByteVectorField(name(), array, fieldType));

        if (hasDocValues && vectorFieldType != null) {
            fieldsToBeAdded.add(new VectorField(name(), array, vectorFieldType));
        }

        if (this.stored) {
            fieldsToBeAdded.add(createStoredFieldForByteVector(name(), array));
        }
        return fieldsToBeAdded;
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
    void updateEngineStats() {
        KNNEngine.LUCENE.setInitialized(true);
    }

    @AllArgsConstructor
    @lombok.Builder
    @Getter
    static class CreateLuceneFieldMapperInput {
        @NonNull
        String name;
        @NonNull
        MultiFields multiFields;
        @NonNull
        CopyTo copyTo;
        @NonNull
        Explicit<Boolean> ignoreMalformed;
        boolean stored;
        boolean hasDocValues;
        KNNMethodContext originalKnnMethodContext;
    }
}
