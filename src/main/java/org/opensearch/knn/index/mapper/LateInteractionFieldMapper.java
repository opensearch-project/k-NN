/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.LateInteractionField;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Field mapper for the late-interaction (multi-vector) field type.
 *
 * <p>Accepts a variable number of equal-dimension vectors per document (a list of lists of numbers,
 * e.g. ColBERT/ColPali token embeddings) and stores them as Lucene {@code BinaryDocValues} via
 * {@link LateInteractionField}. No ANN structure is built; the field is scored in a rescore phase
 * using MaxSim similarity against a query multi-vector.
 *
 * <p>Example mapping:
 * <pre>
 * "tokens": { "type": "late_interaction", "dimension": 128, "space_type": "innerproduct" }
 * </pre>
 *
 * <p>Example document value:
 * <pre>
 * "tokens": [[0.1, 0.2, ...128...], [0.3, 0.4, ...128...], ...]
 * </pre>
 */
public class LateInteractionFieldMapper extends ParametrizedFieldMapper {

    public static final String CONTENT_TYPE = "late_interaction";

    /** Default space type when none is provided. ColBERT MaxSim is an inner-product interaction. */
    static final SpaceType DEFAULT_SPACE_TYPE = SpaceType.INNER_PRODUCT;

    private final int dimension;
    private final SpaceType spaceType;

    protected LateInteractionFieldMapper(
        String simpleName,
        LateInteractionFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        int dimension,
        SpaceType spaceType
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo);
        this.dimension = dimension;
        this.spaceType = spaceType;
    }

    /**
     * Builder for {@link LateInteractionFieldMapper}. Declares the {@code dimension} (required, the
     * per-token dimension) and {@code space_type} (optional) parameters.
     */
    public static class Builder extends ParametrizedFieldMapper.Builder {

        protected final Parameter<Integer> dimension = new Parameter<>(KNNConstants.DIMENSION, false, () -> null, (n, c, o) -> {
            if (o == null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[dimension] is required for field [%s]", n));
            }
            final int value;
            try {
                value = XContentMapValues.nodeIntegerValue(o);
            } catch (Exception e) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Unable to parse [dimension] from [%s] for field [%s]", o, n)
                );
            }
            if (value <= 0) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[dimension] must be > 0 for field [%s]", n));
            }
            return value;
        }, m -> ((LateInteractionFieldMapper) m).dimension);

        protected final Parameter<String> spaceType = Parameter.stringParam(
            KNNConstants.METHOD_PARAMETER_SPACE_TYPE,
            false,
            m -> ((LateInteractionFieldMapper) m).spaceType.getValue(),
            DEFAULT_SPACE_TYPE.getValue()
        ).setValidator(LateInteractionFieldMapper::validateSpaceType);

        protected final Parameter<Map<String, String>> meta = Parameter.metaParam();

        public Builder(String name) {
            super(name);
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return Arrays.asList(dimension, spaceType, meta);
        }

        @Override
        public LateInteractionFieldMapper build(BuilderContext context) {
            if (dimension.getValue() == null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[dimension] is required for field [%s]", name));
            }
            final SpaceType resolvedSpaceType = SpaceType.getSpace(spaceType.getValue());
            final LateInteractionFieldType fieldType = new LateInteractionFieldType(
                buildFullName(context),
                meta.getValue(),
                dimension.getValue(),
                resolvedSpaceType
            );
            return new LateInteractionFieldMapper(
                name,
                fieldType,
                multiFieldsBuilder.build(this, context),
                copyTo.build(),
                dimension.getValue(),
                resolvedSpaceType
            );
        }
    }

    private static void validateSpaceType(String value) {
        final SpaceType spaceType = SpaceType.getSpace(value);
        if (spaceType.getKnnVectorSimilarityFunction() == null) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "space_type [%s] is not supported for late interaction fields", value)
            );
        }
    }

    public static final TypeParser PARSER = new TypeParser((n, c) -> new Builder(n));

    @Override
    public LateInteractionFieldType fieldType() {
        return (LateInteractionFieldType) super.fieldType();
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new Builder(simpleName()).init(this);
    }

    @Override
    public final boolean parsesArrayValue() {
        return true;
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        final float[][] multiVector = parseMultiVector(context);
        if (multiVector == null) {
            return;
        }
        context.doc().add(new LateInteractionField(name(), multiVector));
    }

    /**
     * Parses a list-of-lists value {@code [[..dim..], [..dim..], ...]} into a {@code float[][]}.
     * Every inner vector must have exactly {@link #dimension} elements. Returns {@code null} when the
     * value is a JSON null (nothing to index).
     */
    private float[][] parseMultiVector(ParseContext context) throws IOException {
        final XContentParser parser = context.parser();
        XContentParser.Token token = parser.currentToken();

        if (token == XContentParser.Token.VALUE_NULL) {
            return null;
        }
        if (token != XContentParser.Token.START_ARRAY) {
            throw new MapperParsingException(
                String.format(
                    Locale.ROOT,
                    "Field [%s] of type [%s] expects an array of vectors (list of list of numbers)",
                    name(),
                    CONTENT_TYPE
                )
            );
        }

        final List<float[]> vectors = new ArrayList<>();
        token = parser.nextToken();
        while (token != XContentParser.Token.END_ARRAY) {
            if (token != XContentParser.Token.START_ARRAY) {
                throw new MapperParsingException(
                    String.format(
                        Locale.ROOT,
                        "Field [%s] of type [%s] expects each element to be a vector (array of numbers)",
                        name(),
                        CONTENT_TYPE
                    )
                );
            }
            vectors.add(parseSingleVector(parser));
            token = parser.nextToken();
        }

        if (vectors.isEmpty()) {
            throw new MapperParsingException(
                String.format(Locale.ROOT, "Field [%s] of type [%s] must contain at least one vector", name(), CONTENT_TYPE)
            );
        }
        return vectors.toArray(new float[0][]);
    }

    private float[] parseSingleVector(XContentParser parser) throws IOException {
        final List<Float> values = new ArrayList<>(dimension);
        XContentParser.Token token = parser.nextToken();
        while (token != XContentParser.Token.END_ARRAY) {
            values.add(parser.floatValue());
            token = parser.nextToken();
        }
        if (values.size() != dimension) {
            throw new MapperParsingException(
                String.format(
                    Locale.ROOT,
                    "Field [%s] of type [%s] expects vectors of dimension [%d] but got [%d]",
                    name(),
                    CONTENT_TYPE,
                    dimension,
                    values.size()
                )
            );
        }
        final float[] vector = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            vector[i] = values.get(i);
        }
        return vector;
    }
}
