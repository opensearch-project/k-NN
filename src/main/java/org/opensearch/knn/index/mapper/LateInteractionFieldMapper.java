/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.LateInteractionField;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;


import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Mapper for indexing a late interaction multi-vector field. 
 * Expects a float multi-vector field, and maps it to {@link LateInteractionField} in Lucene.
 */
public class LateInteractionFieldMapper extends KNNVectorFieldMapper {

    private final PerDimensionValidator perDimensionValidator;

    public static LateInteractionFieldMapper createFieldMapper(
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
        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
            fullname,
            metaValue,
            knnMethodConfigContext.getVectorDataType(),
            knnMethodConfigContext::getDimension
        );
        return new LateInteractionFieldMapper(
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
    }

    private LateInteractionFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
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
        // Late Interaction Multi-Vector doesn't use the knn vector codec in Lucene yet.
        this.useLuceneBasedVectorField = false;
        this.perDimensionValidator = PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
        this.fieldType = LateInteractionField.TYPE;
    }

    @Override
    protected VectorValidator getVectorValidator() {
        return VectorValidator.NOOP_VECTOR_VALIDATOR;
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        return perDimensionValidator;
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        return PerDimensionProcessor.NOOP_PROCESSOR;
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        // only invoked for multi-vector fields
        if (VectorDataType.FLOAT != fieldType().getVectorDataType()) {
            throw new IllegalArgumentException(
                "Late Interaction Multi-Vectors are only supported for float vector data type, found: " + fieldType().getVectorDataType());
        }
        validatePreparse();
        final int dimension = fieldType().getKnnMappingConfig().getDimension();
        Optional<List<float[]>> fieldValue = getFloatMultiVectorFromContext(context, dimension);
        
        if (fieldValue.isEmpty()) {
            return;
        }

        final List<float[]> multiVector = fieldValue.get();
        for (float[] v: multiVector) {
            getVectorValidator().validateVector(v);
            getVectorTransformer().transform(v);
        }

        // TODO: we can avoid a double copy here by changing Lucene to accept a List
        context.doc().addAll(getFieldsForMultiVector(multiVector.toArray(new float[0][])));

    }

    protected List<Field> getFieldsForMultiVector(float[][] multiVector) {
        final List<Field> fields = new ArrayList<>();
        fields.add(new LateInteractionField(name(), multiVector));
        // if (this.stored) {
            // fields.add(new StoredField(name(), LateInteractionField.encode(multiVector)));
        // }
        return fields;
    }

    Optional<List<float[]>> getFloatMultiVectorFromContext(ParseContext context, int dimension) throws IOException {
        context.path().add(simpleName());
        PerDimensionValidator perDimensionValidator = getPerDimensionValidator();
        PerDimensionProcessor perDimensionProcessor = getPerDimensionProcessor();

        ArrayList<float[]> multiVector = new ArrayList<>();
        XContentParser.Token token = context.parser().currentToken();
        float value;
        if (token == XContentParser.Token.START_ARRAY) {
            token = context.parser().nextToken();
            while (token == XContentParser.Token.START_ARRAY) {
                token = context.parser().nextToken();
                float[] vector  = new float[dimension];
                int i = 0;
                while (token != XContentParser.Token.END_ARRAY) {
                    value = perDimensionProcessor.process(context.parser().floatValue());
                    perDimensionValidator.validate(value);
                    vector[i++] = value;
                    token = context.parser().nextToken();
                }
                if (i != dimension) {
                    throw new IllegalArgumentException(
                        "Dimension mismatch for composing vectors in provided multi-vector. Expected: " + dimension + ", Found: " + i);
                }
                token = context.parser().nextToken();
            }
            if (token != XContentParser.Token.END_ARRAY) {
                throw new IllegalArgumentException("Malformed input to multi-vector field. Nested array of floats expected.");
            }
        }

        context.path().remove();
        if (multiVector.isEmpty() == true) {
            return Optional.empty();
        }
        // TODO: assert for tests, can remove later
        for (float[] v: multiVector) {
            assert v.length == dimension: "Nested vector dimension=" + v.length + " does not match configured dimension=" + dimension;
        }
        return Optional.of(multiVector);
    }
}
