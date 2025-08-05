/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.KNNExactQueryBuilder;

import java.io.IOException;

import static org.opensearch.core.xcontent.ToXContent.EMPTY_PARAMS;
import static org.opensearch.index.query.AbstractQueryBuilder.BOOST_FIELD;
import static org.opensearch.knn.index.query.KNNExactQueryBuilder.*;

public class KNNExactQueryBuilderParserTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final Float BOOST = 10.5f;
    private static final String SPACE_TYPE = "innerproduct";
    private static final float[] QUERY_VECTOR = { 1.0f, 2.0f, 3.0f, 4.0f };

    public void testFromXContent_Basic() throws IOException {
        KNNExactQueryBuilder knnExactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnExactQueryBuilder.fieldName());
        builder.field(KNNExactQueryBuilder.VECTOR_FIELD.getPreferredName(), knnExactQueryBuilder.vector());
        builder.field(KNNExactQueryBuilder.SPACE_TYPE_FIELD.getPreferredName(), knnExactQueryBuilder.getSpaceType());
        builder.endObject();
        builder.endObject();

        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNExactQueryBuilder actualBuilder = KNNExactQueryBuilderParser.fromXContent(contentParser);

        assertEquals(knnExactQueryBuilder, actualBuilder);
    }

    public void testFromXContent_WithExpandNested() throws IOException {
        KNNExactQueryBuilder knnExactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .expandNested(true)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnExactQueryBuilder.fieldName());
        builder.field(KNNExactQueryBuilder.VECTOR_FIELD.getPreferredName(), knnExactQueryBuilder.vector());
        builder.field(KNNExactQueryBuilder.SPACE_TYPE_FIELD.getPreferredName(), knnExactQueryBuilder.getSpaceType());
        builder.field(KNNExactQueryBuilder.EXPAND_NESTED_FIELD.getPreferredName(), knnExactQueryBuilder.getExpandNested());
        builder.endObject();
        builder.endObject();

        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNExactQueryBuilder actualBuilder = KNNExactQueryBuilderParser.fromXContent(contentParser);

        assertEquals(knnExactQueryBuilder, actualBuilder);
    }

    public void testFromXContent_WithIgnoreUnmapped() throws IOException {
        KNNExactQueryBuilder knnExactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .ignoreUnmapped(true)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnExactQueryBuilder.fieldName());
        builder.field(KNNExactQueryBuilder.VECTOR_FIELD.getPreferredName(), knnExactQueryBuilder.vector());
        builder.field(KNNExactQueryBuilder.SPACE_TYPE_FIELD.getPreferredName(), knnExactQueryBuilder.getSpaceType());
        builder.field(KNNExactQueryBuilder.IGNORE_UNMAPPED_FIELD.getPreferredName(), knnExactQueryBuilder.isIgnoreUnmapped());
        builder.endObject();
        builder.endObject();

        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNExactQueryBuilder actualBuilder = KNNExactQueryBuilderParser.fromXContent(contentParser);

        assertEquals(knnExactQueryBuilder, actualBuilder);
    }

    public void testFromXContent_whenMultiFields_thenException() throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(FIELD_NAME + "1");
        builder.field(KNNExactQueryBuilder.VECTOR_FIELD.getPreferredName(), QUERY_VECTOR);
        builder.endObject();
        builder.startObject(FIELD_NAME + "2");
        builder.field(KNNExactQueryBuilder.VECTOR_FIELD.getPreferredName(), QUERY_VECTOR);
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        Exception exception = expectThrows(ParsingException.class, () -> KNNExactQueryBuilderParser.fromXContent(contentParser));
        assertTrue(exception.getMessage(), exception.getMessage().contains("[exact_knn] query doesn't support multiple fields"));
    }

    public void testToXContent_BoostOnly_thenSucceed() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(NAME);
        builder.startObject(FIELD_NAME);
        builder.field(KNNExactQueryBuilder.VECTOR_FIELD.getPreferredName(), QUERY_VECTOR);
        builder.field(BOOST_FIELD.getPreferredName(), BOOST);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        KNNExactQueryBuilder knnExactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .boost(BOOST)
            .build();
        XContentBuilder testBuilder = XContentFactory.jsonBuilder();
        testBuilder.startObject();
        KNNExactQueryBuilderParser.toXContent(testBuilder, EMPTY_PARAMS, knnExactQueryBuilder);
        testBuilder.endObject();
        assertEquals(builder.toString(), testBuilder.toString());
    }

    public void testToXContent_BoostSpaceTypeNested_thenSucceed() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(NAME);
        builder.startObject(FIELD_NAME);
        builder.field(KNNExactQueryBuilder.VECTOR_FIELD.getPreferredName(), QUERY_VECTOR);
        builder.field(SPACE_TYPE_FIELD.getPreferredName(), SPACE_TYPE);
        builder.field(EXPAND_NESTED_FIELD.getPreferredName(), true);
        builder.field(BOOST_FIELD.getPreferredName(), BOOST);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        KNNExactQueryBuilder knnExactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .boost(BOOST)
            .spaceType(SPACE_TYPE)
            .expandNested(true)
            .build();
        XContentBuilder testBuilder = XContentFactory.jsonBuilder();
        testBuilder.startObject();
        KNNExactQueryBuilderParser.toXContent(testBuilder, EMPTY_PARAMS, knnExactQueryBuilder);
        testBuilder.endObject();
        assertEquals(builder.toString(), testBuilder.toString());
    }
}
