/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.extern.log4j.Log4j2;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ObjectParser;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.KNNExactQueryBuilder;
import org.opensearch.knn.index.query.KNNBuilderAndParserUtils;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

import static org.opensearch.index.query.AbstractQueryBuilder.BOOST_FIELD;
import static org.opensearch.index.query.AbstractQueryBuilder.NAME_FIELD;
import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.index.query.KNNExactQueryBuilder.NAME;
import static org.opensearch.knn.index.query.KNNExactQueryBuilder.VECTOR_FIELD;
import static org.opensearch.knn.index.query.KNNExactQueryBuilder.SPACE_TYPE_FIELD;
import static org.opensearch.knn.index.query.KNNExactQueryBuilder.EXPAND_NESTED_FIELD;
import static org.opensearch.knn.index.query.KNNExactQueryBuilder.IGNORE_UNMAPPED_FIELD;
import static org.opensearch.knn.index.util.IndexUtil.isClusterOnOrAfterMinRequiredVersion;

/**
 * Helper class for parsing and reverse parsing KNNExactQueryBuilder
 */
@Log4j2
public final class KNNExactQueryBuilderParser {

    private static final ObjectParser<KNNExactQueryBuilder.Builder, Void> INTERNAL_PARSER = createInternalObjectParser();

    private static ObjectParser<KNNExactQueryBuilder.Builder, Void> createInternalObjectParser() {
        ObjectParser<KNNExactQueryBuilder.Builder, Void> internalParser = new ObjectParser<>(NAME, KNNExactQueryBuilder.Builder::new);
        internalParser.declareFloat(KNNExactQueryBuilder.Builder::boost, BOOST_FIELD);
        internalParser.declareString(KNNExactQueryBuilder.Builder::queryName, NAME_FIELD);
        internalParser.declareFloatArray((b, v) -> b.vector(KNNBuilderAndParserUtils.floatListToFloatArray(v, NAME)), VECTOR_FIELD);
        internalParser.declareString(KNNExactQueryBuilder.Builder::spaceType, SPACE_TYPE_FIELD);
        internalParser.declareBoolean((b, v) -> {
            if (isClusterOnOrAfterMinRequiredVersion("ignore_unmapped")) {
                b.ignoreUnmapped(v);
            }
        }, IGNORE_UNMAPPED_FIELD);
        internalParser.declareBoolean(KNNExactQueryBuilder.Builder::expandNested, EXPAND_NESTED_FIELD);
        return internalParser;
    }

    /**
     * Stream input for KNNExactQueryBuilder
     *
     * @param in stream out
     * @param minClusterVersionCheck function to check min version
     * @return KNNExactQueryBuilder.Builder class
     * @throws IOException on stream failure
     */
    public static KNNExactQueryBuilder.Builder streamInput(StreamInput in, Function<String, Boolean> minClusterVersionCheck)
        throws IOException {
        KNNExactQueryBuilder.Builder builder = new KNNExactQueryBuilder.Builder();
        builder.fieldName(in.readString());
        builder.vector(in.readFloatArray());
        builder.spaceType(in.readOptionalString());

        if (minClusterVersionCheck.apply("ignore_unmapped")) {
            builder.ignoreUnmapped(in.readOptionalBoolean());
        }

        if (minClusterVersionCheck.apply(EXPAND_NESTED)) {
            builder.expandNested(in.readOptionalBoolean());
        }

        return builder;
    }

    /**
     * Stream output for KNNExactQueryBuilder
     *
     * @param out stream out
     * @param builder KNNExactQueryBuilder to stream
     * @param minClusterVersionCheck function to check min version
     * @throws IOException on stream failure
     */
    public static void streamOutput(StreamOutput out, KNNExactQueryBuilder builder, Function<String, Boolean> minClusterVersionCheck)
        throws IOException {
        out.writeString(builder.fieldName());
        out.writeFloatArray((float[]) builder.vector());
        out.writeOptionalString(builder.getSpaceType());
        if (minClusterVersionCheck.apply("ignore_unmapped")) {
            out.writeOptionalBoolean(builder.isIgnoreUnmapped());
        }
        if (minClusterVersionCheck.apply(EXPAND_NESTED)) {
            out.writeOptionalBoolean(builder.getExpandNested());
        }
    }

    /**
     * Convert XContent to KNNExactQueryBuilder
     *
     * @param parser input parser
     * @return KNNExactQueryBuilder
     * @throws IOException on parsing failure
     */
    public static KNNExactQueryBuilder fromXContent(XContentParser parser) throws IOException {
        String fieldName = null;
        String currentFieldName = null;
        XContentParser.Token token;
        KNNExactQueryBuilder.Builder builder = null;
        List<Object> vector = null;

        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token == XContentParser.Token.START_OBJECT) {
                KNNBuilderAndParserUtils.throwParsingExceptionOnMultipleFields(
                    parser.getTokenLocation(),
                    fieldName,
                    currentFieldName,
                    NAME
                );
                fieldName = currentFieldName;
                builder = INTERNAL_PARSER.apply(parser, null);
            } else {
                KNNBuilderAndParserUtils.throwParsingExceptionOnMultipleFields(
                    parser.getTokenLocation(),
                    fieldName,
                    parser.currentName(),
                    NAME
                );
                fieldName = parser.currentName();
                vector = parser.list();
            }
        }

        if (builder == null) {
            builder = KNNExactQueryBuilder.builder().vector(KNNBuilderAndParserUtils.objectsToFloats(vector, NAME));
        }
        builder.fieldName(fieldName);
        return builder.build();
    }

    /**
     * Convert KNNExactQueryBuilder to XContent
     *
     * @param builder XContent builder to add KNNExactQueryBuilder
     * @param params ToXContent params
     * @param knnExactQueryBuilder KNNExactQueryBuilder to convert
     * @throws IOException on conversion failure
     */
    public static void toXContent(XContentBuilder builder, ToXContent.Params params, KNNExactQueryBuilder knnExactQueryBuilder)
        throws IOException {
        builder.startObject(NAME);
        builder.startObject(knnExactQueryBuilder.fieldName());

        builder.field(VECTOR_FIELD.getPreferredName(), knnExactQueryBuilder.vector());

        if (knnExactQueryBuilder.getSpaceType() != null) {
            builder.field(SPACE_TYPE_FIELD.getPreferredName(), knnExactQueryBuilder.getSpaceType());
        }
        if (knnExactQueryBuilder.isIgnoreUnmapped()) {
            builder.field(IGNORE_UNMAPPED_FIELD.getPreferredName(), knnExactQueryBuilder.isIgnoreUnmapped());
        }
        if (knnExactQueryBuilder.getExpandNested() != null) {
            builder.field(EXPAND_NESTED_FIELD.getPreferredName(), knnExactQueryBuilder.getExpandNested());
        }
        builder.field(BOOST_FIELD.getPreferredName(), knnExactQueryBuilder.boost());
        if (knnExactQueryBuilder.queryName() != null) {
            builder.field(NAME_FIELD.getPreferredName(), knnExactQueryBuilder.queryName());
        }

        builder.endObject();
        builder.endObject();
    }
}
