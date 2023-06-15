/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Optional;

import static org.apache.lucene.index.VectorValues.MAX_DIMENSIONS;

/**
 * Field mapper for case when Lucene has been set as an engine.
 */
public class LuceneFieldMapper extends KNNVectorFieldMapper {

    private static final int LUCENE_MAX_DIMENSION = MAX_DIMENSIONS;

    /** FieldType used for initializing VectorField, which is used for creating binary doc values. **/
    private final FieldType vectorFieldType;
    private final VectorDataType vectorDataType;

    LuceneFieldMapper(final CreateLuceneFieldMapperInput input) {
        super(
            input.getName(),
            input.getMappedFieldType(),
            input.getMultiFields(),
            input.getCopyTo(),
            input.getIgnoreMalformed(),
            input.isStored(),
            input.isHasDocValues()
        );

        vectorDataType = input.getVectorDataType();
        this.knnMethod = input.getKnnMethodContext();
        final VectorSimilarityFunction vectorSimilarityFunction = this.knnMethod.getSpaceType().getVectorSimilarityFunction();

        final int dimension = input.getMappedFieldType().getDimension();
        if (dimension > LUCENE_MAX_DIMENSION) {
            throw new IllegalArgumentException(
                String.format(
                    "Dimension value cannot be greater than [%s] but got [%s] for vector [%s]",
                    LUCENE_MAX_DIMENSION,
                    dimension,
                    input.getName()
                )
            );
        }

        this.fieldType = vectorDataType.createKnnVectorFieldType(dimension, vectorSimilarityFunction);

        if (this.hasDocValues) {
            this.vectorFieldType = vectorDataType.buildDocValuesFieldType(this.knnMethod.getKnnEngine());
        } else {
            this.vectorFieldType = null;
        }
    }

    @Override
    protected void parseCreateField(ParseContext context, int dimension) throws IOException {

        validateIfKNNPluginEnabled();
        validateIfCircuitBreakerIsNotTriggered();

        if (vectorDataType.equals(VectorDataType.BYTE)) {
            Optional<byte[]> arrayOptional = getBytesFromContext(context, dimension);
            if (arrayOptional.isEmpty()) {
                return;
            }
            final byte[] array = arrayOptional.get();
            KnnByteVectorField point = new KnnByteVectorField(name(), array, fieldType);
            context.doc().add(point);
            if (fieldType.stored()) {
                context.doc().add(new StoredField(name(), point.toString()));
            }
            if (hasDocValues && vectorFieldType != null) {
                context.doc().add(new VectorField(name(), array, vectorFieldType));
            }
        } else {
            Optional<float[]> arrayOptional = getFloatsFromContext(context, dimension);

            if (arrayOptional.isEmpty()) {
                return;
            }
            final float[] array = arrayOptional.get();

            KnnVectorField point = new KnnVectorField(name(), array, fieldType);

            context.doc().add(point);
            if (fieldType.stored()) {
                context.doc().add(new StoredField(name(), point.toString()));
            }

            if (hasDocValues && vectorFieldType != null) {
                context.doc().add(new VectorField(name(), array, vectorFieldType));
            }
        }

        context.path().remove();
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
        KNNVectorFieldType mappedFieldType;
        @NonNull
        MultiFields multiFields;
        @NonNull
        CopyTo copyTo;
        @NonNull
        Explicit<Boolean> ignoreMalformed;
        boolean stored;
        boolean hasDocValues;
        VectorDataType vectorDataType;
        @NonNull
        KNNMethodContext knnMethodContext;
    }
}
