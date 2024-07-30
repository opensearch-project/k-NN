/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.extern.log4j.Log4j2;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ObjectParser;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentLocation;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Function;

import static org.opensearch.index.query.AbstractQueryBuilder.BOOST_FIELD;
import static org.opensearch.index.query.AbstractQueryBuilder.NAME_FIELD;
import static org.opensearch.index.query.AbstractQueryBuilder.parseInnerQueryBuilder;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.index.IndexUtil.isClusterOnOrAfterMinRequiredVersion;
import static org.opensearch.knn.index.query.KNNQueryBuilder.FILTER_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.IGNORE_UNMAPPED_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.K_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.MAX_DISTANCE_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.METHOD_PARAMS_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.MIN_SCORE_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.NAME;
import static org.opensearch.knn.index.query.KNNQueryBuilder.VECTOR_FIELD;

/**
 * Helper class responsible for parsing and reverse parsing KNNQueryBuilder's
 */
@Log4j2
public final class KNNQueryBuilderParser {

    private static final ObjectParser<KNNQueryBuilder.Builder, Void> INTERNAL_PARSER = createInternalObjectParser();

    /**
     * For a k-NN query, we need to parse roughly the following structure into a KNNQueryBuilder:
     *  "my_vector2": {
     *      "vector": [2, 3, 5, 6],
     *      "k": 2,
     *      ...
     *   }
     * to simplify the parsing process, we can define an object parser that will the internal structure after the
     * field name. We cannot unfortunately also parse the field name because it ends up in the same structure
     * as the nested portion. So we need to do that separately.
     */
    private static ObjectParser<KNNQueryBuilder.Builder, Void> createInternalObjectParser() {
        ObjectParser<KNNQueryBuilder.Builder, Void> internalParser = new ObjectParser<>(NAME, KNNQueryBuilder.Builder::new);
        internalParser.declareFloat(KNNQueryBuilder.Builder::boost, BOOST_FIELD);
        internalParser.declareString(KNNQueryBuilder.Builder::queryName, NAME_FIELD);
        internalParser.declareFloatArray((b, v) -> b.vector(floatListToFloatArray(v)), VECTOR_FIELD);
        internalParser.declareInt(KNNQueryBuilder.Builder::k, K_FIELD);
        internalParser.declareBoolean((b, v) -> {
            if (isClusterOnOrAfterMinRequiredVersion("ignore_unmapped")) {
                b.ignoreUnmapped(v);
            }
        }, IGNORE_UNMAPPED_FIELD);
        internalParser.declareFloat(KNNQueryBuilder.Builder::maxDistance, MAX_DISTANCE_FIELD);
        internalParser.declareFloat(KNNQueryBuilder.Builder::minScore, MIN_SCORE_FIELD);

        internalParser.declareObject(
            KNNQueryBuilder.Builder::methodParameters,
            (p, v) -> MethodParametersParser.fromXContent(p),
            METHOD_PARAMS_FIELD
        );
        internalParser.declareObject(KNNQueryBuilder.Builder::filter, (p, v) -> parseInnerQueryBuilder(p), FILTER_FIELD);

        return internalParser;
    }

    /**
     * Stream input for KNNQueryBuilder
     *
     * @param in stream out
     * @param minClusterVersionCheck function to check min version
     * @return KNNQueryBuilder.Builder class
     * @throws IOException on stream failure
     */
    public static KNNQueryBuilder.Builder streamInput(StreamInput in, Function<String, Boolean> minClusterVersionCheck) throws IOException {
        KNNQueryBuilder.Builder builder = new KNNQueryBuilder.Builder();
        builder.fieldName(in.readString());
        builder.vector(in.readFloatArray());
        builder.k(in.readInt());
        builder.filter(in.readOptionalNamedWriteable(QueryBuilder.class));

        if (minClusterVersionCheck.apply("ignore_unmapped")) {
            builder.ignoreUnmapped(in.readOptionalBoolean());
        }
        if (minClusterVersionCheck.apply(KNNConstants.RADIAL_SEARCH_KEY)) {
            builder.maxDistance(in.readOptionalFloat());
        }
        if (minClusterVersionCheck.apply(KNNConstants.RADIAL_SEARCH_KEY)) {
            builder.minScore(in.readOptionalFloat());
        }
        if (minClusterVersionCheck.apply(METHOD_PARAMETER)) {
            builder.methodParameters(MethodParametersParser.streamInput(in, IndexUtil::isClusterOnOrAfterMinRequiredVersion));
        }

        return builder;
    }

    /**
     * Stream output for KNNQueryBuilder
     *
     * @param out stream out
     * @param builder KNNQueryBuilder to stream
     * @param minClusterVersionCheck function to check min version
     * @throws IOException on stream failure
     */
    public static void streamOutput(StreamOutput out, KNNQueryBuilder builder, Function<String, Boolean> minClusterVersionCheck)
        throws IOException {
        out.writeString(builder.fieldName());
        out.writeFloatArray((float[]) builder.vector());
        out.writeInt(builder.getK());
        out.writeOptionalNamedWriteable(builder.getFilter());
        if (minClusterVersionCheck.apply("ignore_unmapped")) {
            out.writeOptionalBoolean(builder.isIgnoreUnmapped());
        }
        if (minClusterVersionCheck.apply(KNNConstants.RADIAL_SEARCH_KEY)) {
            out.writeOptionalFloat(builder.getMaxDistance());
        }
        if (minClusterVersionCheck.apply(KNNConstants.RADIAL_SEARCH_KEY)) {
            out.writeOptionalFloat(builder.getMinScore());
        }
        if (minClusterVersionCheck.apply(METHOD_PARAMETER)) {
            MethodParametersParser.streamOutput(out, builder.getMethodParameters(), IndexUtil::isClusterOnOrAfterMinRequiredVersion);
        }
    }

    /**
     * Convert XContent to KNNQueryBuilder
     *
     * @param parser input parser
     * @return KNNQueryBuilder
     * @throws IOException on parsing failure
     */
    public static KNNQueryBuilder fromXContent(XContentParser parser) throws IOException {
        String fieldName = null;
        String currentFieldName = null;
        XContentParser.Token token;
        KNNQueryBuilder.Builder builder = null;
        List<Object> vector = null;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token == XContentParser.Token.START_OBJECT) {
                throwParsingExceptionOnMultipleFields(parser.getTokenLocation(), fieldName, currentFieldName);
                fieldName = currentFieldName;
                builder = INTERNAL_PARSER.apply(parser, null);
            } else {
                throwParsingExceptionOnMultipleFields(parser.getTokenLocation(), fieldName, parser.currentName());
                fieldName = parser.currentName();
                vector = parser.list();
            }
        }

        if (builder == null) {
            builder = KNNQueryBuilder.builder().vector(objectsToFloats(vector));
        }
        builder.fieldName(fieldName);
        return builder.build();
    }

    /**
     * Convert KNNQueryBuilder to XContent
     *
     * @param builder xcontent builder to add KNNQueryBuilder
     * @param params ToXContent params
     * @param knnQueryBuilder KNNQueryBuilder to convert
     * @throws IOException on conversion failure
     */
    public static void toXContent(XContentBuilder builder, ToXContent.Params params, KNNQueryBuilder knnQueryBuilder) throws IOException {
        builder.startObject(NAME);
        builder.startObject(knnQueryBuilder.fieldName());

        builder.field(VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(K_FIELD.getPreferredName(), knnQueryBuilder.getK());
        if (knnQueryBuilder.getFilter() != null) {
            builder.field(FILTER_FIELD.getPreferredName(), knnQueryBuilder.getFilter());
        }
        if (knnQueryBuilder.getMaxDistance() != null) {
            builder.field(MAX_DISTANCE_FIELD.getPreferredName(), knnQueryBuilder.getMaxDistance());
        }
        if (knnQueryBuilder.isIgnoreUnmapped()) {
            builder.field(IGNORE_UNMAPPED_FIELD.getPreferredName(), knnQueryBuilder.isIgnoreUnmapped());
        }
        if (knnQueryBuilder.getMinScore() != null) {
            builder.field(MIN_SCORE_FIELD.getPreferredName(), knnQueryBuilder.getMinScore());
        }
        if (knnQueryBuilder.getMethodParameters() != null) {
            MethodParametersParser.doXContent(builder, knnQueryBuilder.getMethodParameters());
        }

        builder.field(BOOST_FIELD.getPreferredName(), knnQueryBuilder.boost());
        if (knnQueryBuilder.queryName() != null) {
            builder.field(NAME_FIELD.getPreferredName(), knnQueryBuilder.queryName());
        }

        builder.endObject();
        builder.endObject();
    }

    private static float[] floatListToFloatArray(List<Float> floats) {
        if (Objects.isNull(floats) || floats.isEmpty()) {
            throw new IllegalArgumentException(String.format("[%s] field 'vector' requires to be non-null and non-empty", NAME));
        }
        float[] vec = new float[floats.size()];
        for (int i = 0; i < floats.size(); i++) {
            vec[i] = floats.get(i);
        }
        return vec;
    }

    private static float[] objectsToFloats(List<Object> objs) {
        if (Objects.isNull(objs) || objs.isEmpty()) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "[%s] field 'vector' requires to be non-null and non-empty", NAME)
            );
        }
        float[] vec = new float[objs.size()];
        for (int i = 0; i < objs.size(); i++) {
            if ((objs.get(i) instanceof Number) == false) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] field 'vector' requires to be an array of numbers", NAME)
                );
            }
            vec[i] = ((Number) objs.get(i)).floatValue();
        }
        return vec;
    }

    private static void throwParsingExceptionOnMultipleFields(
        XContentLocation contentLocation,
        String processedFieldName,
        String currentFieldName
    ) {
        if (processedFieldName != null) {
            throw new ParsingException(
                contentLocation,
                "[" + NAME + "] query doesn't support multiple fields, found [" + processedFieldName + "] and [" + currentFieldName + "]"
            );
        }
    }
}
