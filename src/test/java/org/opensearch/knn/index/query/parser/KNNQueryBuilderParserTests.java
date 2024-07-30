/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import org.opensearch.Version;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNClusterUtil;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.plugins.SearchPlugin;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.opensearch.core.xcontent.ToXContent.EMPTY_PARAMS;
import static org.opensearch.index.query.AbstractQueryBuilder.BOOST_FIELD;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;
import static org.opensearch.knn.index.query.KNNQueryBuilder.NAME;
import static org.opensearch.knn.index.query.KNNQueryBuilder.EF_SEARCH_FIELD;

public class KNNQueryBuilderParserTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final int K = 1;
    private static final int EF_SEARCH = 10;
    private static final Map<String, ?> HNSW_METHOD_PARAMS = Map.of("ef_search", EF_SEARCH);
    private static final Float MAX_DISTANCE = 1.0f;
    private static final Float MIN_SCORE = 0.5f;
    private static final Float BOOST = 10.5f;
    private static final TermQueryBuilder TERM_QUERY = QueryBuilders.termQuery("field", "value");

    public void testFromXContent() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).k(K).build();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), knnQueryBuilder.getK());
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilderParser.fromXContent(contentParser);
        assertEquals(knnQueryBuilder, actualBuilder);
    }

    public void testFromXContent_KnnWithMethodParameters() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .methodParameters(HNSW_METHOD_PARAMS)
            .build();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), knnQueryBuilder.getK());
        builder.startObject(org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER);
        builder.field(EF_SEARCH_FIELD.getPreferredName(), EF_SEARCH);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilderParser.fromXContent(contentParser);
        assertEquals(knnQueryBuilder, actualBuilder);
    }

    public void testFromXContent_whenDoRadiusSearch_whenDistanceThreshold_whenMethodParameter_thenSucceed() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .methodParameters(HNSW_METHOD_PARAMS)
            .build();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.MAX_DISTANCE_FIELD.getPreferredName(), knnQueryBuilder.getMaxDistance());
        builder.startObject(org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER);
        builder.field(EF_SEARCH_FIELD.getPreferredName(), EF_SEARCH);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilderParser.fromXContent(contentParser);
        assertEquals(knnQueryBuilder, actualBuilder);
    }

    public void testFromXContent_whenDoRadiusSearch_whenScoreThreshold_whenMethodParameter_thenSucceed() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .minScore(MAX_DISTANCE)
            .methodParameters(HNSW_METHOD_PARAMS)
            .build();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.MIN_SCORE_FIELD.getPreferredName(), knnQueryBuilder.getMinScore());
        builder.startObject(org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER);
        builder.field(EF_SEARCH_FIELD.getPreferredName(), EF_SEARCH);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilderParser.fromXContent(contentParser);
        assertEquals(knnQueryBuilder, actualBuilder);
    }

    public void testFromXContent_withFilter() throws Exception {
        final ClusterService clusterService = mockClusterService(Version.CURRENT);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);

        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .filter(TERM_QUERY)
            .build();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), knnQueryBuilder.getK());
        builder.field(KNNQueryBuilder.FILTER_FIELD.getPreferredName(), knnQueryBuilder.getFilter());
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilderParser.fromXContent(contentParser);
        assertEquals(knnQueryBuilder, actualBuilder);
    }

    public void testFromXContent_KnnWithEfSearch_withFilter() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .filter(TERM_QUERY)
            .methodParameters(HNSW_METHOD_PARAMS)
            .build();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), knnQueryBuilder.getK());
        builder.startObject(org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER);
        builder.field(EF_SEARCH_FIELD.getPreferredName(), EF_SEARCH);
        builder.endObject();
        builder.field(KNNQueryBuilder.FILTER_FIELD.getPreferredName(), knnQueryBuilder.getFilter());
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilderParser.fromXContent(contentParser);
        assertEquals(knnQueryBuilder, actualBuilder);
    }

    public void testFromXContent_whenDoRadiusSearch_whenDistanceThreshold_whenFilter_thenSucceed() throws Exception {
        final ClusterService clusterService = mockClusterService(Version.CURRENT);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);

        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .filter(TERM_QUERY)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.MAX_DISTANCE_FIELD.getPreferredName(), knnQueryBuilder.getMaxDistance());
        builder.field(KNNQueryBuilder.FILTER_FIELD.getPreferredName(), knnQueryBuilder.getFilter());
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilderParser.fromXContent(contentParser);
        assertEquals(knnQueryBuilder, actualBuilder);
    }

    public void testFromXContent_whenDoRadiusSearch_whenScoreThreshold_whenFilter_thenSucceed() throws Exception {
        final ClusterService clusterService = mockClusterService(Version.CURRENT);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);

        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .minScore(MIN_SCORE)
            .filter(TERM_QUERY)
            .build();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(knnQueryBuilder.fieldName());
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), knnQueryBuilder.vector());
        builder.field(KNNQueryBuilder.MIN_SCORE_FIELD.getPreferredName(), knnQueryBuilder.getMinScore());
        builder.field(KNNQueryBuilder.FILTER_FIELD.getPreferredName(), knnQueryBuilder.getFilter());
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        KNNQueryBuilder actualBuilder = KNNQueryBuilderParser.fromXContent(contentParser);
        assertEquals(knnQueryBuilder, actualBuilder);
    }

    public void testFromXContent_InvalidQueryVectorType() throws Exception {
        final ClusterService clusterService = mockClusterService(Version.CURRENT);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);

        List<Object> invalidTypeQueryVector = new ArrayList<>();
        invalidTypeQueryVector.add(1.5);
        invalidTypeQueryVector.add(2.5);
        invalidTypeQueryVector.add("a");
        invalidTypeQueryVector.add(null);

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(FIELD_NAME);
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), invalidTypeQueryVector);
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), K);
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilderParser.fromXContent(contentParser)
        );
        assertTrue(exception.getMessage(), exception.getMessage().contains("[knn] failed to parse field [vector]"));
    }

    public void testFromXContent_whenDoRadiusSearch_whenInputInvalidQueryVectorType_thenException() throws Exception {
        final ClusterService clusterService = mockClusterService(Version.CURRENT);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);

        List<Object> invalidTypeQueryVector = new ArrayList<>();
        invalidTypeQueryVector.add(1.5);
        invalidTypeQueryVector.add(2.5);
        invalidTypeQueryVector.add("a");
        invalidTypeQueryVector.add(null);

        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(FIELD_NAME);
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), invalidTypeQueryVector);
        builder.field(KNNQueryBuilder.MAX_DISTANCE_FIELD.getPreferredName(), MAX_DISTANCE);
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilderParser.fromXContent(contentParser)
        );
        assertTrue(exception.getMessage(), exception.getMessage().contains("[knn] failed to parse field [vector]"));
    }

    public void testFromXContent_missingQueryVector() throws Exception {
        final ClusterService clusterService = mockClusterService(Version.CURRENT);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);

        // Test without vector field
        XContentBuilder builderWithoutVectorField = XContentFactory.jsonBuilder();
        builderWithoutVectorField.startObject();
        builderWithoutVectorField.startObject(FIELD_NAME);
        builderWithoutVectorField.field(KNNQueryBuilder.K_FIELD.getPreferredName(), K);
        builderWithoutVectorField.endObject();
        builderWithoutVectorField.endObject();
        XContentParser contentParserWithoutVectorField = createParser(builderWithoutVectorField);
        contentParserWithoutVectorField.nextToken();
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilderParser.fromXContent(contentParserWithoutVectorField)
        );
        assertTrue(exception.getMessage(), exception.getMessage().contains("[knn] requires query vector"));

        // Test empty vector field
        List<Object> emptyQueryVector = new ArrayList<>();
        XContentBuilder builderWithEmptyVector = XContentFactory.jsonBuilder();
        builderWithEmptyVector.startObject();
        builderWithEmptyVector.startObject(FIELD_NAME);
        builderWithEmptyVector.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), emptyQueryVector);
        builderWithEmptyVector.field(KNNQueryBuilder.K_FIELD.getPreferredName(), K);
        builderWithEmptyVector.endObject();
        builderWithEmptyVector.endObject();
        XContentParser contentParserWithEmptyVector = createParser(builderWithEmptyVector);
        contentParserWithEmptyVector.nextToken();
        exception = expectThrows(IllegalArgumentException.class, () -> KNNQueryBuilderParser.fromXContent(contentParserWithEmptyVector));
        assertTrue(exception.getMessage(), exception.getMessage().contains("[knn] failed to parse field [vector]"));
    }

    public void testFromXContent_whenFlat_thenException() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field(FIELD_NAME, queryVector);
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        Exception exception = expectThrows(IllegalArgumentException.class, () -> KNNQueryBuilderParser.fromXContent(contentParser));
        assertTrue(exception.getMessage(), exception.getMessage().contains("[knn] requires exactly one of k, distance or score to be set"));
    }

    public void testFromXContent_whenMultiFields_thenException() throws Exception {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(FIELD_NAME + "1");
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), queryVector);
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), K);
        builder.endObject();
        builder.startObject(FIELD_NAME + "2");
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), queryVector);
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), K);
        builder.endObject();
        builder.endObject();
        XContentParser contentParser = createParser(builder);
        contentParser.nextToken();
        Exception exception = expectThrows(ParsingException.class, () -> KNNQueryBuilderParser.fromXContent(contentParser));
        assertTrue(exception.getMessage(), exception.getMessage().contains("[knn] query doesn't support multiple fields"));
    }

    public void testToXContent_whenParamsVectorBoostK_thenSucceed() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(NAME);
        builder.startObject(FIELD_NAME);
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), queryVector);
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), K);
        builder.field(BOOST_FIELD.getPreferredName(), BOOST);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).k(K).boost(BOOST).build();
        XContentBuilder testBuilder = XContentFactory.jsonBuilder();
        testBuilder.startObject();
        KNNQueryBuilderParser.toXContent(testBuilder, EMPTY_PARAMS, knnQueryBuilder);
        testBuilder.endObject();
        assertEquals(builder.toString(), testBuilder.toString());
    }

    public void testToXContent_whenFilter_thenSucceed() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(NAME);
        builder.startObject(FIELD_NAME);
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), queryVector);
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), K);
        builder.field(KNNQueryBuilder.FILTER_FIELD.getPreferredName(), TERM_QUERY);
        builder.field(BOOST_FIELD.getPreferredName(), BOOST);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .boost(BOOST)
            .filter(TERM_QUERY)
            .build();
        XContentBuilder testBuilder = XContentFactory.jsonBuilder();
        testBuilder.startObject();
        KNNQueryBuilderParser.toXContent(testBuilder, EMPTY_PARAMS, knnQueryBuilder);
        testBuilder.endObject();
        assertEquals(builder.toString(), testBuilder.toString());
    }

    public void testToXContent_whenMaxDistance_thenSucceed() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(NAME);
        builder.startObject(FIELD_NAME);
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), queryVector);
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), 0);
        builder.field(KNNQueryBuilder.MAX_DISTANCE_FIELD.getPreferredName(), MAX_DISTANCE);
        builder.field(BOOST_FIELD.getPreferredName(), BOOST);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .boost(BOOST)
            .maxDistance(MAX_DISTANCE)
            .build();
        XContentBuilder testBuilder = XContentFactory.jsonBuilder();
        testBuilder.startObject();
        KNNQueryBuilderParser.toXContent(testBuilder, EMPTY_PARAMS, knnQueryBuilder);
        testBuilder.endObject();
        assertEquals(builder.toString(), testBuilder.toString());
    }

    public void testToXContent_whenMethodParams_thenSucceed() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject(NAME);
        builder.startObject(FIELD_NAME);
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), queryVector);
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), K);
        builder.startObject(org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER);
        builder.field(EF_SEARCH_FIELD.getPreferredName(), EF_SEARCH);
        builder.endObject();
        builder.field(BOOST_FIELD.getPreferredName(), BOOST);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .boost(BOOST)
            .k(K)
            .methodParameters(HNSW_METHOD_PARAMS)
            .build();
        XContentBuilder testBuilder = XContentFactory.jsonBuilder();
        testBuilder.startObject();
        KNNQueryBuilderParser.toXContent(testBuilder, EMPTY_PARAMS, knnQueryBuilder);
        testBuilder.endObject();
        logger.info(builder.toString());
        logger.info(testBuilder.toString());
        assertEquals(builder.toString(), testBuilder.toString());
    }

    @Override
    protected NamedXContentRegistry xContentRegistry() {
        List<NamedXContentRegistry.Entry> list = ClusterModule.getNamedXWriteables();
        SearchPlugin.QuerySpec<?> spec = new SearchPlugin.QuerySpec<>(
            TermQueryBuilder.NAME,
            TermQueryBuilder::new,
            TermQueryBuilder::fromXContent
        );
        list.add(new NamedXContentRegistry.Entry(QueryBuilder.class, spec.getName(), (p, c) -> spec.getParser().fromXContent(p)));
        NamedXContentRegistry registry = new NamedXContentRegistry(list);
        return registry;
    }
}
