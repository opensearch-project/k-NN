/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor;

import org.opensearch.action.search.SearchRequest;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.apache.lucene.search.join.ScoreMode;
import org.opensearch.index.query.InnerHitBuilder;
import org.opensearch.index.query.NestedQueryBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.ProcessorGenerationContext;
import org.opensearch.core.tasks.TaskId;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNSourceExcludesProcessorTests extends KNNTestCase {

    private ClusterService clusterService;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        clusterService = mock(ClusterService.class);
    }

    private KNNSourceExcludesProcessor createProcessor(List<InnerHitBuilder> innerHitBuilders) {
        return new KNNSourceExcludesProcessor("test-tag", "test-desc", false, clusterService, innerHitBuilders);
    }

    private KNNSourceExcludesProcessor createProcessor() {
        return createProcessor(Collections.emptyList());
    }

    public void testType() {
        assertEquals("knn_default_excludes", createProcessor().getType());
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

    public void testProcessRequest_innerHitIncludesVectorField_skipsExcludes() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.setFetchSourceContext(new FetchSourceContext(true, new String[] { "nested_obj.vec" }, new String[0]));

        KNNSourceExcludesProcessor processor = createProcessor(List.of(innerHit));
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);

        // Should return unmodified because inner hit explicitly includes the vector field
        assertNull(result.source().fetchSource());
    }

    public void testProcessRequest_innerHitWithoutIncludes_appliesExcludes() {
        mockClusterStateWithMapping(
            "test-index",
            Map.of("properties", Map.of("nested_obj", Map.of("properties", Map.of("vec", Map.of("type", "knn_vector", "dimension", 3)))))
        );

        InnerHitBuilder innerHit = new InnerHitBuilder();

        KNNSourceExcludesProcessor processor = createProcessor(List.of(innerHit));
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        SearchRequest result = processor.processRequest(request);

        FetchSourceContext ctx = result.source().fetchSource();
        assertNotNull(ctx);
        Set<String> excludes = Set.of(ctx.excludes());
        assertTrue(excludes.contains("nested_obj.vec"));

        FetchSourceContext innerCtx = innerHit.getFetchSourceContext();
        assertNotNull(innerCtx);
        Set<String> innerExcludes = Set.of(innerCtx.excludes());
        assertTrue(innerExcludes.contains("nested_obj.vec"));
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

    // Factory tests

    public void testFactory_shouldGenerate_nullSearchRequest_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        ProcessorGenerationContext context = new ProcessorGenerationContext(null);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_withParentTask_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());
        request.setParentTask(new TaskId("node1", 1));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_sourceExplicitlyFalse_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(false));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_sourceExplicitlyTrue_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[0], new String[0])));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_storedFieldsNone_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().storedField("_none_"));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_validRequest_returnsTrue() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertTrue(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_withExistingExcludes_returnsTrue() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[0], new String[] { "some_field" })));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertTrue(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_withIncludes_returnsTrue() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().fetchSource(new FetchSourceContext(true, new String[] { "title" }, new String[0])));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertTrue(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_nestedQueryWithExplicitTrueInnerHit_returnsFalse() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.setFetchSourceContext(new FetchSourceContext(true, new String[0], new String[0]));
        NestedQueryBuilder nestedQuery = new NestedQueryBuilder("nested_path", new MatchAllQueryBuilder(), ScoreMode.None);
        nestedQuery.innerHit(innerHit);

        request.source(new SearchSourceBuilder().query(nestedQuery));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertFalse(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_nestedQueryWithNullInnerHitSource_returnsTrue() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");

        InnerHitBuilder innerHit = new InnerHitBuilder();
        NestedQueryBuilder nestedQuery = new NestedQueryBuilder("nested_path", new MatchAllQueryBuilder(), ScoreMode.None);
        nestedQuery.innerHit(innerHit);

        request.source(new SearchSourceBuilder().query(nestedQuery));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertTrue(factory.shouldGenerate(context));
    }

    public void testFactory_shouldGenerate_nestedQueryWithExcludesInInnerHit_returnsTrue() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");

        InnerHitBuilder innerHit = new InnerHitBuilder();
        innerHit.setFetchSourceContext(new FetchSourceContext(true, new String[0], new String[] { "some_field" }));
        NestedQueryBuilder nestedQuery = new NestedQueryBuilder("nested_path", new MatchAllQueryBuilder(), ScoreMode.None);
        nestedQuery.innerHit(innerHit);

        request.source(new SearchSourceBuilder().query(nestedQuery));

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
        assertTrue(factory.shouldGenerate(context));
    }

    public void testFactory_create_returnsValidProcessor() {
        KNNSourceExcludesProcessor.Factory factory = new KNNSourceExcludesProcessor.Factory(clusterService);
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder());

        ProcessorGenerationContext context = new ProcessorGenerationContext(request);
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
    }
}
