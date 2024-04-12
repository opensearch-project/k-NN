/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import java.io.IOException;
import java.util.Locale;
import java.util.Optional;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.util.KNNEngine;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.addStoredFieldForVectorField;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.buildDocValuesFieldType;

/**
 * Field mapper for case when Lucene has been set as an engine.
 */
public class LuceneFieldMapper extends KNNVectorFieldMapper {

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
            input.isHasDocValues(),
            input.getKnnMethodContext().getMethodComponentContext().getIndexVersion()
        );

        vectorDataType = input.getVectorDataType();
        this.knnMethod = input.getKnnMethodContext();
        final VectorSimilarityFunction vectorSimilarityFunction = this.knnMethod.getSpaceType().getVectorSimilarityFunction();

        final int dimension = input.getMappedFieldType().getDimension();
        if (dimension > KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE)) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Dimension value cannot be greater than [%s] but got [%s] for vector [%s]",
                    KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE),
                    dimension,
                    input.getName()
                )
            );
        }

        this.fieldType = vectorDataType.createKnnVectorFieldType(dimension, vectorSimilarityFunction);

        if (this.hasDocValues) {
            this.vectorFieldType = buildDocValuesFieldType(this.knnMethod.getKnnEngine());
        } else {
            this.vectorFieldType = null;
        }
    }

    @Override
    protected void parseCreateField(ParseContext context, int dimension, SpaceType spaceType, MethodComponentContext methodComponentContext)
        throws IOException {

        validateIfKNNPluginEnabled();
        validateIfCircuitBreakerIsNotTriggered();

        if (VectorDataType.BYTE == vectorDataType) {
            Optional<byte[]> bytesArrayOptional = getBytesFromContext(context, dimension);
            if (bytesArrayOptional.isEmpty()) {
                return;
            }
            final byte[] array = bytesArrayOptional.get();
            spaceType.validateVector(array);
            KnnByteVectorField point = new KnnByteVectorField(name(), array, fieldType);

            context.doc().add(point);
            addStoredFieldForVectorField(context, fieldType, name(), point);

            if (hasDocValues && vectorFieldType != null) {
                context.doc().add(new VectorField(name(), array, vectorFieldType));
            }
        } else if (VectorDataType.FLOAT == vectorDataType) {
            Optional<float[]> floatsArrayOptional = getFloatsFromContext(context, dimension, methodComponentContext);

            if (floatsArrayOptional.isEmpty()) {
                return;
            }
            final float[] array = floatsArrayOptional.get();
            spaceType.validateVector(array);
            KnnVectorField point = new KnnVectorField(name(), array, fieldType);

            context.doc().add(point);
            addStoredFieldForVectorField(context, fieldType, name(), point);

            if (hasDocValues && vectorFieldType != null) {
                context.doc().add(new VectorField(name(), array, vectorFieldType));
            }
        } else {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Cannot parse context for unsupported values provided for field [%s]", VECTOR_DATA_TYPE_FIELD)
            );
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
