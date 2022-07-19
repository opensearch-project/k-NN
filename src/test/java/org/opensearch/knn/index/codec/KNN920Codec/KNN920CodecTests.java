/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN920Codec;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.watcher.ResourceWatcherService;

import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.index.codec.KNNCodecFactory.CodecDelegateFactory.createKNN92DefaultDelegate;

public class KNN920CodecTests extends KNNCodecTestCase {

    public void testMultiFieldsKnnIndex() throws Exception {
        testMultiFieldsKnnIndex(KNN920Codec.builder().delegate(createKNN92DefaultDelegate()).build());
    }

    public void testBuildFromModelTemplate() throws InterruptedException, ExecutionException, IOException {
        testBuildFromModelTemplate((KNN920Codec.builder().delegate(createKNN92DefaultDelegate()).build()));
    }

    public void testKnnVectorIndex() throws Exception {
        MapperService mapperService = mock(MapperService.class);
        final KNN920Codec actualCodec = KNN920Codec.builder()
            .delegate(createKNN92DefaultDelegate())
            .mapperService(Optional.of(mapperService))
            .build();
        final KNN920Codec codec = KNN920Codec.builder()
            .delegate(createKNN92DefaultDelegate())
            .mapperService(Optional.of(mapperService))
            .build();
        setUpMockClusterService();
        Directory dir = newFSDirectory(createTempDir());
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(codec);

        /**
         * Add doc with field "test_vector"
         */
        FieldType luceneFieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        luceneFieldType.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        luceneFieldType.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName());
        luceneFieldType.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        luceneFieldType.putAttribute(KNNConstants.HNSW_ALGO_M, "32");
        luceneFieldType.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "512");
        luceneFieldType.freeze();

        float[] array = { 1.0f, 3.0f, 4.0f };
        VectorField vectorField = new VectorField("test_vector", array, luceneFieldType);
        RandomIndexWriter writer = new RandomIndexWriter(random(), dir, iwc);
        Document doc = new Document();
        doc.add(vectorField);
        writer.addDocument(doc);
        writer.close();

        /**
         * Add doc with field "my_vector"
         */
        IndexWriterConfig iwc1 = newIndexWriterConfig();
        iwc1.setMergeScheduler(new SerialMergeScheduler());
        iwc1.setCodec(actualCodec);
        writer = new RandomIndexWriter(random(), dir, iwc1);
        float[] array1 = { 6.0f, 14.0f };
        VectorField vectorField1 = new VectorField("my_vector", array1, luceneFieldType);
        Document doc1 = new Document();
        doc1.add(vectorField1);
        writer.addDocument(doc1);
        IndexReader reader = writer.getReader();
        writer.close();
        ResourceWatcherService resourceWatcherService = createDisabledResourceWatcherService();
        NativeMemoryLoadStrategy.IndexLoadStrategy.initialize(resourceWatcherService);

        reader.close();
        dir.close();
        resourceWatcherService.close();
        NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance().close();
    }
}
