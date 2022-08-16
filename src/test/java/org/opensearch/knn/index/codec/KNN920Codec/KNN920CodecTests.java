/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN920Codec;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.query.KNNQueryFactory;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.watcher.ResourceWatcherService;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutionException;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.codec.KNNCodecFactory.CodecDelegateFactory.createKNN92DefaultDelegate;

public class KNN920CodecTests extends KNNCodecTestCase {

    public void testMultiFieldsKnnIndex() throws Exception {
        testMultiFieldsKnnIndex(KNN920Codec.builder().delegate(createKNN92DefaultDelegate()).build());
    }

    public void testBuildFromModelTemplate() throws InterruptedException, ExecutionException, IOException {
        testBuildFromModelTemplate((KNN920Codec.builder().delegate(createKNN92DefaultDelegate()).build()));
    }

    public void testKnnVectorIndex() throws Exception {
        final String fieldName = "test_vector";
        final String field1Name = "my_vector";
        final MapperService mapperService = mock(MapperService.class);
        final KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(HNSW_ALGO_M, 16, HNSW_ALGO_EF_CONSTRUCTION, 256))
        );
        final KNNVectorFieldMapper.KNNVectorFieldType mappedFieldType1 = new KNNVectorFieldMapper.KNNVectorFieldType(
            fieldName,
            Map.of(),
            3,
            knnMethodContext
        );
        final KNNVectorFieldMapper.KNNVectorFieldType mappedFieldType2 = new KNNVectorFieldMapper.KNNVectorFieldType(
            field1Name,
            Map.of(),
            2,
            knnMethodContext
        );
        when(mapperService.fieldType(eq(fieldName))).thenReturn(mappedFieldType1);
        when(mapperService.fieldType(eq(field1Name))).thenReturn(mappedFieldType2);

        var knnVectorsFormat = spy(new KNN920PerFieldKnnVectorsFormat(Optional.of(mapperService)));

        final KNN920Codec actualCodec = KNN920Codec.builder()
            .delegate(createKNN92DefaultDelegate())
            .knnVectorsFormat(knnVectorsFormat)
            .build();
        final KNN920Codec codec = KNN920Codec.builder().delegate(createKNN92DefaultDelegate()).knnVectorsFormat(knnVectorsFormat).build();
        setUpMockClusterService();
        Directory dir = newFSDirectory(createTempDir());
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(codec);

        /**
         * Add doc with field "test_vector"
         */
        final FieldType luceneFieldType = KnnVectorField.createFieldType(3, VectorSimilarityFunction.EUCLIDEAN);
        float[] array = { 1.0f, 3.0f, 4.0f };
        KnnVectorField vectorField = new KnnVectorField(fieldName, array, luceneFieldType);
        RandomIndexWriter writer = new RandomIndexWriter(random(), dir, iwc);
        Document doc = new Document();
        doc.add(vectorField);
        writer.addDocument(doc);
        writer.commit();
        IndexReader reader = writer.getReader();
        writer.close();

        verify(knnVectorsFormat).getKnnVectorsFormatForField(anyString());

        IndexSearcher searcher = new IndexSearcher(reader);
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        Query query = KNNQueryFactory.create(
            KNNEngine.LUCENE,
            "dummy",
            fieldName,
            new float[] { 1.0f, 0.0f, 0.0f },
            1,
            null,
            mockQueryShardContext
        );

        assertEquals(1, searcher.count(query));

        reader.close();

        /**
         * Add doc with field "my_vector"
         */
        IndexWriterConfig iwc1 = newIndexWriterConfig();
        iwc1.setMergeScheduler(new SerialMergeScheduler());
        iwc1.setCodec(actualCodec);
        writer = new RandomIndexWriter(random(), dir, iwc1);
        final FieldType luceneFieldType1 = KnnVectorField.createFieldType(2, VectorSimilarityFunction.EUCLIDEAN);
        float[] array1 = { 6.0f, 14.0f };
        KnnVectorField vectorField1 = new KnnVectorField(field1Name, array1, luceneFieldType1);
        Document doc1 = new Document();
        doc1.add(vectorField1);
        writer.addDocument(doc1);
        IndexReader reader1 = writer.getReader();
        writer.close();
        ResourceWatcherService resourceWatcherService = createDisabledResourceWatcherService();
        NativeMemoryLoadStrategy.IndexLoadStrategy.initialize(resourceWatcherService);

        verify(knnVectorsFormat, times(2)).getKnnVectorsFormatForField(anyString());

        IndexSearcher searcher1 = new IndexSearcher(reader1);
        Query query1 = KNNQueryFactory.create(
            KNNEngine.LUCENE,
            "dummy",
            field1Name,
            new float[] { 1.0f, 0.0f },
            1,
            null,
            mockQueryShardContext
        );

        assertEquals(1, searcher1.count(query1));

        reader1.close();
        dir.close();
        resourceWatcherService.close();
        NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance().close();
    }
}
