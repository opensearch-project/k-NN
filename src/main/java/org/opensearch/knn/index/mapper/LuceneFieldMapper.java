/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import com.google.common.collect.ImmutableMap;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.index.SpaceType.COSINESIMIL;
import static org.opensearch.knn.index.SpaceType.INNER_PRODUCT;
import static org.opensearch.knn.index.SpaceType.L2;

/**
 * Field mapper for case when Lucene has been set as an engine.
 */
public class LuceneFieldMapper extends KNNVectorFieldMapper {

    /** FieldType used for initializing VectorField, which is used for creating binary doc values. **/
    private final FieldType vectorFieldType;

    private static final Map<SpaceType, VectorSimilarityFunction> SPACE_TYPE_TO_VECTOR_SIMILARITY_FUNCTION = ImmutableMap.of(
        L2,
        VectorSimilarityFunction.EUCLIDEAN,
        COSINESIMIL,
        VectorSimilarityFunction.COSINE,
        INNER_PRODUCT,
        VectorSimilarityFunction.DOT_PRODUCT
    );

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

        this.knnMethod = input.getKnnMethodContext();
        final SpaceType spaceType = this.knnMethod.getSpaceType();
        final VectorSimilarityFunction vectorSimilarityFunction = Optional.ofNullable(
            SPACE_TYPE_TO_VECTOR_SIMILARITY_FUNCTION.get(spaceType)
        ).orElseThrow(() -> new IllegalArgumentException(String.format("Space type [%s] is not supported for Lucene engine", spaceType)));

        final int dimension = input.getMappedFieldType().getDimension();
        this.fieldType = KnnVectorField.createFieldType(dimension, vectorSimilarityFunction);

        if (this.hasDocValues) {
            this.vectorFieldType = buildDocValuesFieldType(this.knnMethod.getKnnEngine());
        } else {
            this.vectorFieldType = null;
        }
    }

    private static FieldType buildDocValuesFieldType(KNNEngine knnEngine) {
        FieldType field = new FieldType();
        field.putAttribute(KNN_ENGINE, knnEngine.getName());
        field.setDocValuesType(DocValuesType.BINARY);
        field.freeze();
        return field;
    }

    @Override
    protected void parseCreateField(ParseContext context, int dimension) throws IOException {

        validateIfKNNPluginEnabled();
        validateIfCircuitBreakerIsNotTriggered();

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

        context.path().remove();
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
        @NonNull
        KNNMethodContext knnMethodContext;
    }
}
