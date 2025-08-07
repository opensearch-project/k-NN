/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import org.opensearch.Version;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.ExactKNNQueryBuilder;
import org.opensearch.knn.index.util.KNNClusterUtil;

import java.io.IOException;

import static org.opensearch.core.xcontent.ToXContent.EMPTY_PARAMS;
import static org.opensearch.index.query.AbstractQueryBuilder.BOOST_FIELD;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;
import static org.opensearch.knn.index.query.ExactKNNQueryBuilder.NAME;
import static org.opensearch.knn.index.query.ExactKNNQueryBuilder.SPACE_TYPE_FIELD;

public class ExactKNNQueryBuilderParserTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final Float BOOST = 10.5f;
    private static final String SPACE_TYPE = "innerproduct";
    private static final float[] QUERY_VECTOR = { 1.0f, 2.0f, 3.0f, 4.0f };

    public void testFromXContent_Basic() throws IOException {
        ExactKNNQueryBuilder exactKNNQueryBuilder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(exactKNNQueryBuilder.fieldName());
        builder.field(ExactKNNQueryBuilder.VECTOR_FIELD.getPreferredName(), exactKNNQueryBuilder.getVector());
        builder.field(ExactKNNQueryBuilder.SPACE_TYPE_FIELD.getPreferredName(), exactKNNQueryBuilder.getSpaceType());
        builder.endObject();
        builder.endObject();

        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        ExactKNNQueryBuilder actualBuilder = ExactKNNQueryBuilderParser.fromXContent(contentParser);

        assertEquals(exactKNNQueryBuilder, actualBuilder);
    }

    public void testFromXContent_WithIgnoreUnmapped() throws IOException {
        final ClusterService clusterService = mockClusterService(Version.CURRENT);
        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);

        ExactKNNQueryBuilder exactKNNQueryBuilder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .ignoreUnmapped(true)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(exactKNNQueryBuilder.fieldName());
        builder.field(ExactKNNQueryBuilder.VECTOR_FIELD.getPreferredName(), exactKNNQueryBuilder.getVector());
        builder.field(ExactKNNQueryBuilder.SPACE_TYPE_FIELD.getPreferredName(), exactKNNQueryBuilder.getSpaceType());
        builder.field(ExactKNNQueryBuilder.IGNORE_UNMAPPED_FIELD.getPreferredName(), exactKNNQueryBuilder.isIgnoreUnmapped());
        builder.endObject();
        builder.endObject();

        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        ExactKNNQueryBuilder actualBuilder = ExactKNNQueryBuilderParser.fromXContent(contentParser);

        assertEquals(exactKNNQueryBuilder, actualBuilder);
    }

    public void testFromXContent_whenMultiFields_thenException() throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(FIELD_NAME + "1");
        builder.field(ExactKNNQueryBuilder.VECTOR_FIELD.getPreferredName(), QUERY_VECTOR);
        builder.endObject();
        builder.startObject(FIELD_NAME + "2");
        builder.field(ExactKNNQueryBuilder.VECTOR_FIELD.getPreferredName(), QUERY_VECTOR);
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        Exception exception = expectThrows(ParsingException.class, () -> ExactKNNQueryBuilderParser.fromXContent(contentParser));
        assertTrue(exception.getMessage(), exception.getMessage().contains("[exact_knn] query doesn't support multiple fields"));
    }

    public void testToXContent_BoostOnly_thenSucceed() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(NAME);
        builder.startObject(FIELD_NAME);
        builder.field(ExactKNNQueryBuilder.VECTOR_FIELD.getPreferredName(), QUERY_VECTOR);
        builder.field(BOOST_FIELD.getPreferredName(), BOOST);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        ExactKNNQueryBuilder exactKNNQueryBuilder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .boost(BOOST)
            .build();
        XContentBuilder testBuilder = XContentFactory.jsonBuilder();
        testBuilder.startObject();
        ExactKNNQueryBuilderParser.toXContent(testBuilder, EMPTY_PARAMS, exactKNNQueryBuilder);
        testBuilder.endObject();
        assertEquals(builder.toString(), testBuilder.toString());
    }

    public void testToXContent_BoostSpaceType_thenSucceed() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(NAME);
        builder.startObject(FIELD_NAME);
        builder.field(ExactKNNQueryBuilder.VECTOR_FIELD.getPreferredName(), QUERY_VECTOR);
        builder.field(SPACE_TYPE_FIELD.getPreferredName(), SPACE_TYPE);
        builder.field(BOOST_FIELD.getPreferredName(), BOOST);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        ExactKNNQueryBuilder exactKNNQueryBuilder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .boost(BOOST)
            .spaceType(SPACE_TYPE)
            .build();
        XContentBuilder testBuilder = XContentFactory.jsonBuilder();
        testBuilder.startObject();
        ExactKNNQueryBuilderParser.toXContent(testBuilder, EMPTY_PARAMS, exactKNNQueryBuilder);
        testBuilder.endObject();
        assertEquals(builder.toString(), testBuilder.toString());
    }
}
