/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor;

import org.opensearch.action.search.SearchRequest;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.apache.lucene.search.join.ScoreMode;
import org.opensearch.index.query.InnerHitBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.NestedQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.ProcessorGenerationContext;
import org.opensearch.core.tasks.TaskId;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNSourceExcludesProcessorTests extends KNNTestCase {

    private ClusterService clusterService;
    private IndexNameExpressionResolver indexNameExpressionResolver;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        clusterService = mock(ClusterService.class);
        indexNameExpressionResolver = mock(IndexNameExpressionResolver.class);
    }

    private KNNSourceExcludesProcessor createProcessor() {
        return new KNNSourceExcludesProcessor("test-tag", "test-desc", false, clusterService, indexNameExpressionResolver);
    }

    private SearchRequest requestWithNestedQuery(String index, InnerHitBuilder innerHit) {
        NestedQueryBuilder nestedQuery = new NestedQueryBuilder("nested_obj", new MatchAllQueryBuilder(), ScoreMode.None);
        nestedQuery.innerHit(innerHit);
        SearchRequest request = new SearchRequest(index);
        request.source(new SearchSourceBuilder().query(nestedQuery));
        return request;
    }

    public void testType() {
        assertEquals("knn_default_excludes", createProcessor().getType());
    }

    public void testGetExecutionStage_returnsPreUserDefined() {
        assertEquals(KNNSourceExcludesProcessor.ExecutionStage.PRE_USER_DEFINED, createProcessor().getExecutionStage());
    }

    public void testProcessRequest_noIndices_returnsUnmodified() {
        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest();
        request.source(new SearchSourceBuilder());
        mockClusterState(Map.of());

        SearchRequest result = processor.processRequest(request);
        assertNull(result.source().fetchSource());
    }

    public void testProcessRequest_indexWithVectorField_addsExclude() {
        mockClusterStateWithMapping("test-index", Map.of("properties", Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 3))));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        assertTrue(ctx.fetchSource());
        assertEquals(0, ctx.includes().length);
        assertArrayEquals(new String[] { "my_vector" }, ctx.excludes());
    }

    public void testProcessRequest_preservesUserIncludes() {
        mockClusterStateWithMapping("test-index", Map.of("properties", Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 2))));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[] { "title" }, new String[0])));

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertArrayEquals(new String[] { "title" }, ctx.includes());
        assertArrayEquals(new String[] { "my_vector" }, ctx.excludes());
    }

    public void testProcessRequest_userIncludesVectorField_noExclude() {
        mockClusterStateWithMapping("test-index", Map.of("properties", Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 2))));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[] { "my_vector" }, new String[0])));

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertArrayEquals(new String[] { "my_vector" }, ctx.includes());
        assertEquals(0, ctx.excludes().length);
    }

    public void testProcessRequest_userIncludesGlobCoveringVectorField_noExclude() {
        mockClusterStateWithMapping("test-index", Map.of("properties", Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 2))));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[] { "my_*" }, new String[0])));

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertArrayEquals(new String[] { "my_*" }, ctx.includes());
        assertEquals(0, ctx.excludes().length);
    }

    public void testProcessRequest_userExcludesGlobCoveringVectorField_notDuplicated() {
        mockClusterStateWithMapping("test-index", Map.of("properties", Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 2))));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[0], new String[] { "my_*" })));

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertArrayEquals(new String[] { "my_*" }, ctx.excludes());
        assertEquals(1, ctx.excludes().length);
    }

    public void testProcessRequest_userExcludesDotStarPatternDoesNotCoverVectorField_addsLiteral() {
        // "my_vector.*" does NOT match "my_vector" — Regex.simpleMatch requires at least a dot+char suffix.
        // The processor correctly adds "my_vector" as a literal exclude alongside the user pattern.
        mockClusterStateWithMapping("test-index", Map.of("properties", Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 2))));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[0], new String[] { "my_vector.*" })));

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        Set<String> excludes = Set.of(ctx.excludes());
        assertTrue("my_vector.*  does not cover my_vector — literal should be added", excludes.contains("my_vector"));
        assertTrue("User pattern should be preserved", excludes.contains("my_vector.*"));
        assertEquals(2, ctx.excludes().length);
    }

    public void testProcessRequest_preservesExistingExcludes() {
        mockClusterStateWithMapping("test-index", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 1))));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(
            new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[0], new String[] { "internal_field" }))
        );

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        Set<String> excludes = Set.of(ctx.excludes());
        assertTrue(excludes.contains("internal_field"));
        assertTrue(excludes.contains("vec"));
    }

    public void testProcessRequest_noVectorFields_returnsUnmodified() {
        mockClusterStateWithMapping("test-index", Map.of("properties", Map.of("title", Map.of("type", "text"))));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);
        assertNull(result.source().fetchSource());
    }

    public void testProcessRequest_multipleIndices_collectsAllVectorFields() {
        IndexMetadata idx1 = mockIndexMetadata(Map.of("properties", Map.of("vec1", Map.of("type", "knn_vector", "dimension", 3))));
        IndexMetadata idx2 = mockIndexMetadata(Map.of("properties", Map.of("vec2", Map.of("type", "knn_vector", "dimension", 5))));
        mockClusterState(Map.of("index-1", idx1, "index-2", idx2));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("index-1", "index-2");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        Set<String> excludes = Set.of(ctx.excludes());
        assertTrue(excludes.contains("vec1"));
        assertTrue(excludes.contains("vec2"));
    }

    public void testProcessRequest_aliasResolvesToConcreteIndex_addsExclude() {
        IndexMetadata idx = mockIndexMetadata(Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3))));
        ClusterState state = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        when(clusterService.state()).thenReturn(state);
        when(state.metadata()).thenReturn(metadata);
        when(metadata.index("concrete-index")).thenReturn(idx);
        when(indexNameExpressionResolver.concreteIndexNames(any(ClusterState.class), any(SearchRequest.class))).thenReturn(
            new String[] { "concrete-index" }
        );

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("my-alias");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        assertArrayEquals(new String[] { "vec" }, ctx.excludes());
    }

    public void testProcessRequest_innerHitExplicitTrueSource_skipsExcludes() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.setFetchSourceContext(new FetchSourceContext(true, new String[0], new String[0]));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest result = processor.processRequest(requestWithNestedQuery("test-index", innerHit));

        assertNull("Top-level excludes should not be added when inner hit has explicit true _source", result.source().fetchSource());

        // Inner hit source context should remain exactly as set — {true, [], []}
        FetchSourceContext innerCtx = innerHit.getFetchSourceContext();
        assertNotNull(innerCtx);
        assertTrue(innerCtx.fetchSource());
        assertEquals(0, innerCtx.includes().length);
        assertEquals(0, innerCtx.excludes().length);
    }

    public void testProcessRequest_innerHitSourceIncludesVectorField_skipsExcludes() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.setFetchSourceContext(new FetchSourceContext(true, new String[] { "nested_obj.vec" }, new String[0]));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest result = processor.processRequest(requestWithNestedQuery("test-index", innerHit));

        // Early return — top-level source unchanged
        assertNull(result.source().fetchSource());

        // Inner hit source context unchanged — includes still has the vector field
        FetchSourceContext innerCtx = innerHit.getFetchSourceContext();
        assertNotNull(innerCtx);
        assertTrue(innerCtx.fetchSource());
        assertArrayEquals(new String[] { "nested_obj.vec" }, innerCtx.includes());
        assertEquals(0, innerCtx.excludes().length);
    }

    public void testProcessRequest_innerHitFetchFieldsIncludesVectorField_skipsExcludes() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.addFetchField("nested_obj.vec");

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest result = processor.processRequest(requestWithNestedQuery("test-index", innerHit));

        // Early return — top-level source unchanged
        assertNull(result.source().fetchSource());

        // Inner hit source context was null before and remains null — only fetchFields was set
        assertNull(innerHit.getFetchSourceContext());
    }

    public void testProcessRequest_innerHitFetchFieldsWithoutVectorField_appliesExcludes() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.addFetchField("nested_obj.name");

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest result = processor.processRequest(requestWithNestedQuery("test-index", innerHit));

        // Top-level gets vec excluded
        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        assertTrue(Set.of(ctx.excludes()).contains("nested_obj.vec"));

        // Inner hit source context also gets vec excluded
        FetchSourceContext innerCtx = innerHit.getFetchSourceContext();
        assertNotNull(innerCtx);
        assertTrue(Set.of(innerCtx.excludes()).contains("nested_obj.vec"));
    }

    public void testProcessRequest_innerHitExcludesVectorField_notDuplicated() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.setFetchSourceContext(new FetchSourceContext(true, new String[0], new String[] { "nested_obj.vec" }));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest result = processor.processRequest(requestWithNestedQuery("test-index", innerHit));

        // Top-level also gets vec excluded
        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        assertTrue(Set.of(ctx.excludes()).contains("nested_obj.vec"));

        // Inner hit already had vec in excludes — should not be duplicated
        FetchSourceContext innerCtx = innerHit.getFetchSourceContext();
        assertNotNull(innerCtx);
        Set<String> innerExcludes = Set.of(innerCtx.excludes());
        assertTrue(innerExcludes.contains("nested_obj.vec"));
        assertEquals("vec should appear exactly once in inner hit excludes", 1, innerCtx.excludes().length);
    }

    public void testProcessRequest_innerHitWithoutIncludes_appliesExcludes() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest result = processor.processRequest(requestWithNestedQuery("test-index", innerHit));

        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        assertTrue(Set.of(ctx.excludes()).contains("nested_obj.vec"));

        FetchSourceContext innerCtx = innerHit.getFetchSourceContext();
        assertNotNull(innerCtx);
        assertTrue(Set.of(innerCtx.excludes()).contains("nested_obj.vec"));
    }

    public void testProcessRequest_innerHitSourceFalse_appliesTopLevelExcludesButNotInnerHit() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.setFetchSourceContext(new FetchSourceContext(false));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest result = processor.processRequest(requestWithNestedQuery("test-index", innerHit));

        // Top-level excludes are applied
        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        assertTrue(Set.of(ctx.excludes()).contains("nested_obj.vec"));

        // Inner hit source remains false — not flipped to true with excludes
        FetchSourceContext innerCtx = innerHit.getFetchSourceContext();
        assertNotNull(innerCtx);
        assertFalse("Inner hit _source should remain false", innerCtx.fetchSource());
        assertEquals(0, innerCtx.excludes().length);
    }

    public void testProcessRequest_multipleInnerHits_firstSkipsTriggers_noExcludes() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        // First inner hit explicitly requests the vector — should cause entire processor to skip
        InnerHitBuilder innerHit1 = new InnerHitBuilder();
        innerHit1.setFetchSourceContext(new FetchSourceContext(true, new String[] { "nested_obj.vec" }, new String[0]));

        // Second inner hit has no constraints — would otherwise get excludes applied
        InnerHitBuilder innerHit2 = new InnerHitBuilder();

        NestedQueryBuilder nestedQuery1 = new NestedQueryBuilder("nested_obj", new MatchAllQueryBuilder(), ScoreMode.None);
        nestedQuery1.innerHit(innerHit1);
        NestedQueryBuilder nestedQuery2 = new NestedQueryBuilder("nested_obj", new MatchAllQueryBuilder(), ScoreMode.None);
        nestedQuery2.innerHit(innerHit2);

        SearchRequest request = new SearchRequest("test-index");
        request.source(
            new SearchSourceBuilder().query(new org.opensearch.index.query.BoolQueryBuilder().should(nestedQuery1).should(nestedQuery2))
        );

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest result = processor.processRequest(request);

        // No excludes added at top level since first inner hit requests vector
        assertNull(result.source().fetchSource());
        // Second inner hit also unchanged — apply loop never ran
        assertNull(innerHit2.getFetchSourceContext());
    }

    public void testProcessRequest_multiIndexOneSourceDisabled_onlyEnabledIndexVectorsExcluded() {
        IndexMetadata enabledIdx = mockIndexMetadata(Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3))));
        IndexMetadata disabledIdx = mockIndexMetadata(
            Map.of("_source", Map.of("enabled", false), "properties", Map.of("vec2", Map.of("type", "knn_vector", "dimension", 3)))
        );
        mockClusterState(Map.of("enabled-index", enabledIdx, "disabled-index", disabledIdx));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("enabled-index", "disabled-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        Set<String> excludes = Set.of(ctx.excludes());
        assertTrue("vec from enabled index should be excluded", excludes.contains("vec"));
        assertFalse("vec2 from source-disabled index should not be added", excludes.contains("vec2"));
    }

    public void testCollectVectorFields_nestedObject() {
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("nested_obj", Map.of("properties", Map.of("embedding", Map.of("type", "knn_vector", "dimension", 4))))
        );

        Set<String> fields = new HashSet<>();
        KNNSourceExcludesProcessor.collectVectorFields(mapping, "", fields);

        assertTrue(fields.contains("nested_obj.embedding"));
        assertEquals(1, fields.size());
    }

    public void testCollectVectorFields_multipleFields() {
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of(
                "vec1",
                Map.of("type", "knn_vector", "dimension", 2),
                "vec2",
                Map.of("type", "knn_vector", "dimension", 3),
                "title",
                Map.of("type", "text")
            )
        );

        Set<String> fields = new HashSet<>();
        KNNSourceExcludesProcessor.collectVectorFields(mapping, "", fields);

        assertTrue(fields.contains("vec1"));
        assertTrue(fields.contains("vec2"));
        assertEquals(2, fields.size());
    }

    public void testCollectVectorFields_deeplyNestedObject() {
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of(
                "level1",
                Map.of(
                    "properties",
                    Map.of("level2", Map.of("properties", Map.of("deep_vec", Map.of("type", "knn_vector", "dimension", 8))))
                )
            )
        );

        Set<String> fields = new HashSet<>();
        KNNSourceExcludesProcessor.collectVectorFields(mapping, "", fields);

        assertTrue(fields.contains("level1.level2.deep_vec"));
        assertEquals(1, fields.size());
    }

    public void testCollectVectorFields_emptyProperties() {
        Map<String, Object> mapping = Map.of("properties", Map.of());

        Set<String> fields = new HashSet<>();
        KNNSourceExcludesProcessor.collectVectorFields(mapping, "", fields);

        assertTrue(fields.isEmpty());
    }

    public void testCollectVectorFields_noPropertiesKey() {
        Map<String, Object> mapping = Map.of("some_other_key", "value");

        Set<String> fields = new HashSet<>();
        KNNSourceExcludesProcessor.collectVectorFields(mapping, "", fields);

        assertTrue(fields.isEmpty());
    }

    public void testProcessRequest_sourceDisabledAtIndexLevel_returnsUnmodified() {
        IndexMetadata indexMetadata = mockIndexMetadata(
            Map.of("_source", Map.of("enabled", false), "properties", Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 3)))
        );
        mockClusterState(Map.of("test-index", indexMetadata));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);
        assertNull("No excludes should be added when _source is disabled at index level", result.source().fetchSource());
    }

    public void testProcessRequest_vectorFieldAlreadyExcludedByMappingLiteral_notDuplicated() {
        IndexMetadata indexMetadata = mockIndexMetadata(
            Map.of(
                "_source",
                Map.of("excludes", java.util.List.of("my_vector")),
                "properties",
                Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 3))
            )
        );
        mockClusterState(Map.of("test-index", indexMetadata));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);
        assertNull(
            "No request-level excludes should be added when vector field is already excluded by index mapping",
            result.source().fetchSource()
        );
    }

    public void testProcessRequest_vectorFieldAlreadyExcludedByMappingGlob_notDuplicated() {
        IndexMetadata indexMetadata = mockIndexMetadata(
            Map.of(
                "_source",
                Map.of("excludes", java.util.List.of("my_*")),
                "properties",
                Map.of("my_vector", Map.of("type", "knn_vector", "dimension", 3))
            )
        );
        mockClusterState(Map.of("test-index", indexMetadata));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);
        assertNull(
            "No request-level excludes should be added when vector field is already covered by a glob in index mapping",
            result.source().fetchSource()
        );
    }

    public void testProcessRequest_partialMappingExcludes_onlyAddsUncoveredVectorFields() {
        IndexMetadata indexMetadata = mockIndexMetadata(
            Map.of(
                "_source",
                Map.of("excludes", java.util.List.of("vec1")),
                "properties",
                Map.of("vec1", Map.of("type", "knn_vector", "dimension", 3), "vec2", Map.of("type", "knn_vector", "dimension", 3))
            )
        );
        mockClusterState(Map.of("test-index", indexMetadata));

        KNNSourceExcludesProcessor processor = createProcessor();
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);
        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        Set<String> excludes = Set.of(ctx.excludes());
        assertFalse("vec1 is already excluded by mapping, should not be duplicated", excludes.contains("vec1"));
        assertTrue("vec2 is not excluded by mapping and should be added", excludes.contains("vec2"));
    }

    // Factory tests

    public void testFactory_shouldGenerate_nullSearchRequest_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        ProcessorGenerationContext context = new ProcessorGenerationContext(null, null);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_withParentTask_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());
        request.setParentTask(new TaskId("node1", 1));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request, null);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_sourceExplicitlyFalse_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(false));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request, null);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_sourceExplicitlyTrue_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[0], new String[0])));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request, null);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_storedFieldsNone_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().storedField("_none_"));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request, null);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_validRequest_returnsTrue() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        ProcessorGenerationContext context = new ProcessorGenerationContext(request, null);
        assertTrue(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_withExistingExcludes_returnsTrue() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[0], new String[] { "some_field" })));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request, null);
        assertTrue(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_withIncludes_returnsTrue() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[] { "title" }, new String[0])));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request, null);
        assertTrue(factory.shouldGenerate(context));
    }

    public void testFactory_create_returnsValidProcessor() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService, indexNameExpressionResolver);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        ProcessorGenerationContext context = new ProcessorGenerationContext(request, null);
        factory.shouldGenerate(context);

        var processor = factory.create(Map.of(), "tag", "desc", false, Map.of(), null);
        assertNotNull(processor);
        assertEquals("knn_default_excludes", processor.getType());
    }

    // Helper methods

    @SuppressWarnings("unchecked")
    private void mockClusterStateWithMapping(String indexName, Map<String, Object> mappingSource) {
        IndexMetadata indexMetadata = mockIndexMetadata(mappingSource);
        mockClusterState(Map.of(indexName, indexMetadata));
    }

    private IndexMetadata mockIndexMetadata(Map<String, Object> mappingSource) {
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.sourceAsMap()).thenReturn(mappingSource);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        return indexMetadata;
    }

    private void mockClusterState(Map<String, IndexMetadata> indices) {
        ClusterState state = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        when(clusterService.state()).thenReturn(state);
        when(state.metadata()).thenReturn(metadata);
        for (Map.Entry<String, IndexMetadata> entry : indices.entrySet()) {
            when(metadata.index(entry.getKey())).thenReturn(entry.getValue());
        }
        when(indexNameExpressionResolver.concreteIndexNames(any(ClusterState.class), any(SearchRequest.class))).thenReturn(
            indices.keySet().toArray(new String[0])
        );
    }
}
