/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.fetch;

import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexService;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.fieldvisitor.FieldsVisitor;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.search.SearchHit;
import org.opensearch.search.fetch.FetchContext;
import org.opensearch.search.fetch.FetchSubPhase;
import org.opensearch.search.fetch.FetchSubPhaseProcessor;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;

import static java.util.Collections.emptyMap;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_SYNTHETIC_SOURCE_ENABLED_SETTING;

public class KNNFetchSubPhaseTests extends KNNSingleNodeTestCase {
    static final String testIndexName = "test-index";
    static final String fieldName = "test-field-1";

    public void testSyntheticSourceSettingDisabled() throws IOException {
        FetchContext fetchContext = mock(FetchContext.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNN_SYNTHETIC_SOURCE_ENABLED_SETTING)).thenReturn(false);
        when(fetchContext.getIndexSettings()).thenReturn(indexSettings);
        KNNFetchSubPhase phase = new KNNFetchSubPhase();
        FetchSubPhaseProcessor processor = phase.getProcessor(fetchContext);
        assertNull(processor);
    }

    public void testKNNFetchSubPhaseGetProcessor() throws IOException, ExecutionException, InterruptedException {
        XContentBuilder mapping = constructMappingBuilder();
        IndexService indexService = createIndex(testIndexName, constructSettings(), "_doc", mapping);
        addKnnDoc(testIndexName, "1", fieldName, new Float[] { 2.5F, 3.5F });

        IndexSettings indexSettings = indexService.getIndexSettings();
        MapperService mapperService = indexService.mapperService();
        FetchContext fetchContext = mock(FetchContext.class);
        when(fetchContext.mapperService()).thenReturn(mapperService);
        when(fetchContext.getIndexSettings()).thenReturn(indexSettings);

        QueryShardContext queryShardContext = indexService.newQueryShardContext(0, null, System::currentTimeMillis, null);
        SearchLookup searchLookup = queryShardContext.newFetchLookup();
        when(fetchContext.searchLookup()).thenReturn(searchLookup);

        KNNFetchSubPhase phase = new KNNFetchSubPhase();
        FetchSubPhaseProcessor processor = phase.getProcessor(fetchContext);
        assertNotNull(processor);
        assertTrue(processor instanceof KNNFetchSubPhase.KNNFetchSubPhaseProcessor);
        KNNFetchSubPhase.KNNFetchSubPhaseProcessor fetchProcessor = (KNNFetchSubPhase.KNNFetchSubPhaseProcessor) processor;
        assertNotNull(fetchProcessor.getFields());
        assertEquals(fetchProcessor.getFields().get(0).getField(), fieldName);
    }

    public void testKNNFetchSubPhaseProcessorProcessValue() throws IOException, ExecutionException, InterruptedException {
        XContentBuilder mapping = constructMappingBuilder();
        IndexService indexService = createIndex(testIndexName, constructSettings(), "_doc", mapping);
        addKnnDoc(testIndexName, "1", fieldName, new Float[] { 2.5F, 3.5F });

        IndexSettings indexSettings = indexService.getIndexSettings();
        MapperService mapperService = indexService.mapperService();
        FetchContext fetchContext = mock(FetchContext.class);
        when(fetchContext.mapperService()).thenReturn(mapperService);
        when(fetchContext.getIndexSettings()).thenReturn(indexSettings);

        IndexShard indexShard = indexService.getShard(0);
        Engine.Searcher searcher = indexShard.acquireSearcher("Test");
        QueryShardContext queryShardContext = indexService.newQueryShardContext(0, searcher, System::currentTimeMillis, null);
        SearchLookup searchLookup = queryShardContext.newFetchLookup();
        when(fetchContext.searchLookup()).thenReturn(searchLookup);

        KNNFetchSubPhase phase = new KNNFetchSubPhase();
        FetchSubPhaseProcessor processor = phase.getProcessor(fetchContext);

        List<LeafReaderContext> listLeafReadContext = queryShardContext.getIndexReader().leaves();
        LeafReaderContext leafReaderContext = listLeafReadContext.get(0);
        FieldsVisitor fieldsVisitor = new FieldsVisitor(true);
        leafReaderContext.reader().storedFields().document(0, fieldsVisitor);

        final SearchHit searchHit = new SearchHit(0, "1", Collections.emptyMap(), emptyMap());

        FetchSubPhase.HitContext hitContext = new FetchSubPhase.HitContext(searchHit, leafReaderContext, 0, searchLookup.source());
        BytesReference bytesReference = fieldsVisitor.source();
        hitContext.sourceLookup().setSource(bytesReference);
        hitContext.hit().sourceRef(bytesReference);

        String preSource = hitContext.hit().getSourceAsString();
        assertNotNull(preSource);
        assertFalse(preSource.contains("test-field-1"));
        processor.setNextReader(leafReaderContext);
        processor.process(hitContext);
        String afterSource = hitContext.hit().getSourceAsString();
        assertTrue(afterSource.contains("\"test-field-1\":[2.5,3.5]"));
        searcher.close();
    }

    private Settings constructSettings() throws IOException {
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", true)
            .put("index.knn", true)
            .build();
        return indexSettingWithSynthetic;
    }

    private XContentBuilder constructMappingBuilder() throws IOException {
        Integer dimension = 2;

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L2;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .startArray("excludes")
            .value(fieldName)
            .endArray()
            .endObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return builder;
    }
}
