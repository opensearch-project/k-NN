/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.mockito.ArgumentCaptor;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.search.extension.MMRSearchExtBuilder;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.PipelineProcessingContext;
import org.opensearch.search.pipeline.SystemGeneratedProcessor;
import org.opensearch.transport.client.Client;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.opensearch.knn.common.KNNConstants.*;

public class MMROverSampleProcessorTests extends MMRTestCase {
    private Client mockClient;
    private MMROverSampleProcessor processor;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mockClient = mock(Client.class);
        processor = new MMROverSampleProcessor("testTag", true, mockClient, getMockMMRQueryTransformers());
    }

    public void testMetadata() {
        assertEquals(MMROverSampleProcessor.TYPE, processor.getType());
        assertTrue(processor.getDescription().contains("system generated processor"));
        assertEquals("testTag", processor.getTag());
        assertTrue(processor.isIgnoreFailure());
        assertEquals(SystemGeneratedProcessor.ExecutionStage.POST_USER_DEFINED, processor.getExecutionStage());
    }

    public void testSynchronousProcessRequestThrows() {
        UnsupportedOperationException exception = assertThrows(
            UnsupportedOperationException.class,
            () -> processor.processRequest(new SearchRequest())
        );
        String expectedError = "Should not try to use mmr_over_sample to process a search request synchronously.";
        assertEquals(expectedError, exception.getMessage());
    }

    public void testProcessRequestAsync_nullRequest_callsOnFailure() {
        ActionListener<SearchRequest> listener = mock(ActionListener.class);

        processor.processRequestAsync(null, new PipelineProcessingContext(), listener);

        String expectedError = "Search request passed to mmr_over_sample search request processor must have mmr search extension.";
        verifyException(listener, IllegalStateException.class, expectedError);
    }

    public void testExtractMMRExtension_whenMissing_thenException() {
        ActionListener<SearchRequest> listener = mock(ActionListener.class);
        SearchRequest request = new SearchRequest();
        request.source(new SearchSourceBuilder().ext(Collections.emptyList())); // no extensions

        processor.processRequestAsync(request, new PipelineProcessingContext(), listener);

        String expectedError = "SearchRequest passed to mmr_over_sample processor must have an MMRSearchExtBuilder";
        verifyException(listener, IllegalStateException.class, expectedError);
    }

    public void testProcessRequestAsync_whenHappyCase() {
        String indexName = "test-index";
        PipelineProcessingContext pipelineProcessingContext = new PipelineProcessingContext();
        ActionListener<SearchRequest> listener = mock(ActionListener.class);

        SearchRequest request = buildSearchRequest(new String[] { indexName }, new MMRSearchExtBuilder.Builder().build());

        mockClusterIndexMetadata(Map.of(indexName, Collections.emptyMap()));

        processor.processRequestAsync(request, pipelineProcessingContext, listener);

        ArgumentCaptor<SearchRequest> captor = ArgumentCaptor.forClass(SearchRequest.class);
        verify(listener).onResponse(captor.capture());
        SearchRequest searchRequest = captor.getValue();
        assertEquals(30, searchRequest.source().size());
        MMRRerankContext mmrRerankContext = (MMRRerankContext) pipelineProcessingContext.getAttribute(MMR_RERANK_CONTEXT);
        assertEquals(10, (int) mmrRerankContext.getOriginalQuerySize());
        assertEquals(0.5f, mmrRerankContext.getDiversity(), DELTA);
    }

    public void testProcessRequestAsync_whenNullQueryBuilder_thenException() {
        String indexName = "test-index";
        PipelineProcessingContext pipelineProcessingContext = new PipelineProcessingContext();
        ActionListener<SearchRequest> listener = mock(ActionListener.class);

        SearchRequest request = buildSearchRequest(new String[] { indexName }, new MMRSearchExtBuilder.Builder().build());
        request.source().query(null);

        mockClusterIndexMetadata(Map.of(indexName, Collections.emptyMap()));

        processor.processRequestAsync(request, pipelineProcessingContext, listener);

        String expectedError = "Query builder must not be null to do Maximal Marginal Relevance rerank.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testProcessRequestAsync_whenUnsupportedQueryBuilder_thenException() {
        String indexName = "test-index";
        PipelineProcessingContext pipelineProcessingContext = new PipelineProcessingContext();
        ActionListener<SearchRequest> listener = mock(ActionListener.class);

        SearchRequest request = buildSearchRequest(new String[] { indexName }, new MMRSearchExtBuilder.Builder().build());
        BoolQueryBuilder boolQueryBuilder = new BoolQueryBuilder();
        request.source().query(boolQueryBuilder);

        mockClusterIndexMetadata(Map.of(indexName, Collections.emptyMap()));

        processor.processRequestAsync(request, pipelineProcessingContext, listener);

        String expectedError = "Maximal Marginal Relevance rerank doesn't support the query type [bool]";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testProcessRequestAsync_whenHappyCaseWithRemoteIndex() {
        String indexName = "test-index";
        String remoteIndexName = "remote:test-index";
        String vectorFieldName = "vectorField";
        PipelineProcessingContext pipelineProcessingContext = new PipelineProcessingContext();
        ActionListener<SearchRequest> listener = mock(ActionListener.class);

        MMRSearchExtBuilder mmrSearchExtBuilder = new MMRSearchExtBuilder.Builder().vectorFieldPath(vectorFieldName)
            .spaceType(SpaceType.L2.getValue())
            .vectorFieldDataType(VectorDataType.FLOAT.getValue())
            .build();
        SearchRequest request = buildSearchRequest(new String[] { indexName, remoteIndexName }, mmrSearchExtBuilder);

        mockClusterIndexMetadata(
            Map.of(
                indexName,
                Map.of(
                    "properties",
                    Map.of(
                        vectorFieldName,
                        Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                    )
                )
            )
        );

        processor.processRequestAsync(request, pipelineProcessingContext, listener);

        ArgumentCaptor<SearchRequest> captor = ArgumentCaptor.forClass(SearchRequest.class);
        verify(listener).onResponse(captor.capture());
        SearchRequest searchRequest = captor.getValue();
        assertEquals(30, searchRequest.source().size());
        MMRRerankContext mmrRerankContext = (MMRRerankContext) pipelineProcessingContext.getAttribute(MMR_RERANK_CONTEXT);
        assertEquals(10, (int) mmrRerankContext.getOriginalQuerySize());
        assertEquals(0.5f, mmrRerankContext.getDiversity(), DELTA);
        assertEquals(vectorFieldName, mmrRerankContext.getVectorFieldPath());
        assertEquals(SpaceType.L2, mmrRerankContext.getSpaceType());
        assertEquals(VectorDataType.FLOAT, mmrRerankContext.getVectorDataType());
    }

    public void testProcessRequestAsync_whenRemoteIndexWithoutSpaceType_thenException() {
        String remoteIndexName = "remote:test-index";
        String vectorFieldName = "vectorField";
        PipelineProcessingContext pipelineProcessingContext = new PipelineProcessingContext();
        ActionListener<SearchRequest> listener = mock(ActionListener.class);

        MMRSearchExtBuilder mmrSearchExtBuilder = new MMRSearchExtBuilder.Builder().vectorFieldPath(vectorFieldName).build();
        SearchRequest request = buildSearchRequest(new String[] { remoteIndexName }, mmrSearchExtBuilder);

        mockClusterIndexMetadata(Collections.emptyMap());

        processor.processRequestAsync(request, pipelineProcessingContext, listener);

        String expectedError =
            "vector_field_space_type is required in the MMR query extension when querying remote indices [remote:test-index].";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testProcessRequestAsync_whenRemoteIndexWithoutVectorDataType_thenException() {
        String remoteIndexName = "remote:test-index";
        String vectorFieldName = "vectorField";
        PipelineProcessingContext pipelineProcessingContext = new PipelineProcessingContext();
        ActionListener<SearchRequest> listener = mock(ActionListener.class);

        MMRSearchExtBuilder mmrSearchExtBuilder = new MMRSearchExtBuilder.Builder().vectorFieldPath(vectorFieldName)
            .spaceType(SpaceType.L2.getValue())
            .build();
        SearchRequest request = buildSearchRequest(new String[] { remoteIndexName }, mmrSearchExtBuilder);

        mockClusterIndexMetadata(Collections.emptyMap());

        processor.processRequestAsync(request, pipelineProcessingContext, listener);

        String expectedError =
            "vector_field_data_type is required in the MMR query extension when querying remote indices [remote:test-index].";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testProcessRequestAsync_whenKnnFieldWithModelId() {
        String indexName = "test-index";
        String vectorFieldName = "vectorField";
        String modelId = "modelId";
        PipelineProcessingContext pipelineProcessingContext = new PipelineProcessingContext();
        ActionListener<SearchRequest> listener = mock(ActionListener.class);

        MMRSearchExtBuilder mmrSearchExtBuilder = new MMRSearchExtBuilder.Builder().vectorFieldPath(vectorFieldName)
            .spaceType(SpaceType.L2.getValue())
            .build();
        SearchRequest request = buildSearchRequest(new String[] { indexName }, mmrSearchExtBuilder);
        FetchSourceContext fetchSourceContext = new FetchSourceContext(true, new String[] {}, new String[] { vectorFieldName });
        request.source().fetchSource(fetchSourceContext);

        mockClusterIndexMetadata(
            Map.of(
                indexName,
                Map.of("properties", Map.of(vectorFieldName, Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId)))
            )
        );
        MMRVectorFieldInfo vectorFieldInfo = new MMRVectorFieldInfo();
        vectorFieldInfo.setVectorDataType(VectorDataType.FLOAT);
        vectorFieldInfo.setSpaceType(SpaceType.L2);
        mockModelMetadata(mockClient, Map.of(modelId, vectorFieldInfo));

        processor.processRequestAsync(request, pipelineProcessingContext, listener);

        ArgumentCaptor<SearchRequest> captor = ArgumentCaptor.forClass(SearchRequest.class);
        verify(listener).onResponse(captor.capture());
        SearchRequest searchRequest = captor.getValue();
        assertEquals(30, searchRequest.source().size());
        assertEquals("Fetch source should be set to fetch all fields.", 0, searchRequest.source().fetchSource().excludes().length);
        MMRRerankContext mmrRerankContext = (MMRRerankContext) pipelineProcessingContext.getAttribute(MMR_RERANK_CONTEXT);
        assertEquals(10, (int) mmrRerankContext.getOriginalQuerySize());
        assertEquals(0.5f, mmrRerankContext.getDiversity(), DELTA);
        assertEquals(vectorFieldName, mmrRerankContext.getVectorFieldPath());
        assertEquals(SpaceType.L2, mmrRerankContext.getSpaceType());
        assertEquals(VectorDataType.FLOAT, mmrRerankContext.getVectorDataType());
        assertEquals(fetchSourceContext, mmrRerankContext.getOriginalFetchSourceContext());
    }

    private Map<String, MMRQueryTransformer<? extends QueryBuilder>> getMockMMRQueryTransformers() {
        MMRQueryTransformer<KNNQueryBuilder> transformer = mock(MMRKnnQueryTransformer.class);
        // mock a no-op knn query transformer here
        doAnswer(invocation -> {
            ActionListener<Void> listener = invocation.getArgument(1);
            // Simulate success
            listener.onResponse(null);
            return null; // void method must return null
        }).when(transformer).transform(any(KNNQueryBuilder.class), any(ActionListener.class), any(MMRTransformContext.class));
        return Map.of(KNNQueryBuilder.NAME, transformer);
    }

    private SearchRequest buildSearchRequest(String[] indices, MMRSearchExtBuilder mmrSearchExtBuilder) {
        KNNQueryBuilder queryBuilder = mock(KNNQueryBuilder.class);
        when(queryBuilder.getWriteableName()).thenReturn(KNNQueryBuilder.NAME);

        SearchRequest request = new SearchRequest();
        request.indices(indices);

        SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
        searchSourceBuilder.query(queryBuilder);
        searchSourceBuilder.ext(List.of(mmrSearchExtBuilder));
        request.source(searchSourceBuilder);

        return request;
    }
}
