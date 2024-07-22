/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.opensearch.Version;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN87Codec.KNN87Codec;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.KNNSettings.MODEL_CACHE_SIZE_LIMIT_SETTING;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.assertBinaryIndexLoadableByEngine;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.assertFileInCorrectLocation;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.assertLoadableByEngine;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.assertValidFooter;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.getRandomVectors;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.RandomVectorDocValuesProducer;

public class KNN80DocValuesConsumerTests extends KNNTestCase {

    private static final int EF_SEARCH = 10;
    private static final Map<String, ?> HNSW_METHODPARAMETERS = Map.of(METHOD_PARAMETER_EF_SEARCH, EF_SEARCH);

    private static Directory directory;
    private static Codec codec;

    @BeforeClass
    public static void setStaticVariables() {
        directory = newFSDirectory(createTempDir());
        codec = new KNN87Codec();
    }

    @AfterClass
    public static void closeStaticVariables() throws IOException {
        directory.close();
    }

    public void testAddBinaryField_withKNN() throws IOException {
        // Confirm that addKNNBinaryField will get called if the k-NN parameter is true
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test-field")
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .build();
        DocValuesProducer docValuesProducer = mock(DocValuesProducer.class);

        DocValuesConsumer delegate = mock(DocValuesConsumer.class);
        doNothing().when(delegate).addBinaryField(fieldInfo, docValuesProducer);

        final boolean[] called = { false };
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(delegate, null) {

            @Override
            public void addKNNBinaryField(FieldInfo field, DocValuesProducer valuesProducer, boolean isMerge, boolean isRefresh) {
                called[0] = true;
            }
        };

        knn80DocValuesConsumer.addBinaryField(fieldInfo, docValuesProducer);

        verify(delegate, times(1)).addBinaryField(fieldInfo, docValuesProducer);
        assertTrue(called[0]);
    }

    public void testAddBinaryField_withoutKNN() throws IOException {
        // Confirm that the KNN80DocValuesConsumer will just call delegate AddBinaryField when k-NN parameter is
        // not set
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test-field").build();
        DocValuesProducer docValuesProducer = mock(DocValuesProducer.class);

        DocValuesConsumer delegate = mock(DocValuesConsumer.class);
        doNothing().when(delegate).addBinaryField(fieldInfo, docValuesProducer);

        String segmentName = String.format("test_segment%s", randomAlphaOfLength(4));
        int docsInSegment = 100;

        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        FieldInfos fieldInfos = mock(FieldInfos.class);
        SegmentWriteState state = new SegmentWriteState(null, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        final boolean[] called = { false };
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(delegate, state) {

            @Override
            public void addKNNBinaryField(FieldInfo field, DocValuesProducer valuesProducer, boolean isMerge, boolean isRefresh) {
                called[0] = true;
            }
        };

        knn80DocValuesConsumer.addBinaryField(fieldInfo, docValuesProducer);

        verify(delegate, times(1)).addBinaryField(fieldInfo, docValuesProducer);
        assertFalse(called[0]);
    }

    public void testAddKNNBinaryField_noVectors() throws IOException {
        // When there are no new vectors, no more graph index requests should be added
        RandomVectorDocValuesProducer randomVectorDocValuesProducer = new RandomVectorDocValuesProducer(0, 128);
        Long initialGraphIndexRequests = KNNCounter.GRAPH_INDEX_REQUESTS.getCount();
        Long initialRefreshOperations = KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue();
        Long initialMergeOperations = KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue();
        Long initialMergeSize = KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue();
        Long initialMergeDocs = KNNGraphValue.MERGE_TOTAL_DOCS.getValue();
        String segmentName = String.format("test_segment%s", randomAlphaOfLength(4));
        int docsInSegment = 100;

        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        FieldInfos fieldInfos = mock(FieldInfos.class);
        SegmentWriteState state = new SegmentWriteState(null, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(null, state);
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test-field").build();
        knn80DocValuesConsumer.addKNNBinaryField(fieldInfo, randomVectorDocValuesProducer, true, true);
        assertEquals(initialGraphIndexRequests, KNNCounter.GRAPH_INDEX_REQUESTS.getCount());
        assertEquals(initialRefreshOperations, KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        assertEquals(initialMergeOperations, KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        assertEquals(initialMergeSize, KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());
        assertEquals(initialMergeDocs, KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
    }

    public void testAddKNNBinaryField_fromScratch_nmslibCurrent() throws IOException {
        // Set information about the segment and the fields
        String segmentName = String.format("test_segment%s", randomAlphaOfLength(4));
        int docsInSegment = 100;
        String fieldName = String.format("test_field%s", randomAlphaOfLength(4));

        KNNEngine knnEngine = KNNEngine.NMSLIB;
        SpaceType spaceType = SpaceType.COSINESIMIL;
        int dimension = 16;

        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            knnEngine,
            spaceType,
            new MethodComponentContext(METHOD_HNSW, ImmutableMap.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 512))
        );

        String parameterString = XContentFactory.jsonBuilder().map(knnEngine.getMethodAsMap(knnMethodContext)).toString();

        FieldInfo[] fieldInfoArray = new FieldInfo[] {
            KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName)
                .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
                .addAttribute(KNNConstants.KNN_ENGINE, knnEngine.getName())
                .addAttribute(KNNConstants.SPACE_TYPE, spaceType.getValue())
                .addAttribute(KNNConstants.PARAMETERS, parameterString)
                .build() };

        FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        SegmentWriteState state = new SegmentWriteState(null, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        long initialRefreshOperations = KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue();
        long initialMergeOperations = KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue();

        // Add documents to the field
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(null, state);
        RandomVectorDocValuesProducer randomVectorDocValuesProducer = new RandomVectorDocValuesProducer(docsInSegment, dimension);
        knn80DocValuesConsumer.addKNNBinaryField(fieldInfoArray[0], randomVectorDocValuesProducer, true, true);

        // The document should be created in the correct location
        String expectedFile = KNNCodecUtil.buildEngineFileName(segmentName, knnEngine.getVersion(), fieldName, knnEngine.getExtension());
        assertFileInCorrectLocation(state, expectedFile);

        // The footer should be valid
        assertValidFooter(state.directory, expectedFile);

        // The document should be readable by nmslib
        assertLoadableByEngine(null, state, expectedFile, knnEngine, spaceType, dimension);

        // The graph creation statistics should be updated
        assertEquals(1 + initialRefreshOperations, (long) KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        assertEquals(1 + initialMergeOperations, (long) KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());
    }

    public void testAddKNNBinaryField_fromScratch_nmslibLegacy() throws IOException {
        // Set information about the segment and the fields
        String segmentName = String.format("test_segment%s", randomAlphaOfLength(4));
        int docsInSegment = 100;
        String fieldName = String.format("test_field%s", randomAlphaOfLength(4));

        KNNEngine knnEngine = KNNEngine.NMSLIB;
        SpaceType spaceType = SpaceType.COSINESIMIL;
        int dimension = 16;

        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        FieldInfo[] fieldInfoArray = new FieldInfo[] {
            KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName)
                .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
                .addAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "512")
                .addAttribute(KNNConstants.HNSW_ALGO_M, "16")
                .addAttribute(KNNConstants.SPACE_TYPE, spaceType.getValue())
                .build() };

        FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        SegmentWriteState state = new SegmentWriteState(null, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        long initialRefreshOperations = KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue();
        long initialMergeOperations = KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue();

        // Add documents to the field
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(null, state);
        RandomVectorDocValuesProducer randomVectorDocValuesProducer = new RandomVectorDocValuesProducer(docsInSegment, dimension);
        knn80DocValuesConsumer.addKNNBinaryField(fieldInfoArray[0], randomVectorDocValuesProducer, true, true);

        // The document should be created in the correct location
        String expectedFile = KNNCodecUtil.buildEngineFileName(segmentName, knnEngine.getVersion(), fieldName, knnEngine.getExtension());
        assertFileInCorrectLocation(state, expectedFile);

        // The footer should be valid
        assertValidFooter(state.directory, expectedFile);

        // The document should be readable by nmslib
        assertLoadableByEngine(null, state, expectedFile, knnEngine, spaceType, dimension);

        // The graph creation statistics should be updated
        assertEquals(1 + initialRefreshOperations, (long) KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        assertEquals(1 + initialMergeOperations, (long) KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());
    }

    public void testAddKNNBinaryField_fromScratch_faissCurrent() throws IOException {
        String segmentName = String.format("test_segment%s", randomAlphaOfLength(4));
        int docsInSegment = 100;
        String fieldName = String.format("test_field%s", randomAlphaOfLength(4));

        KNNEngine knnEngine = KNNEngine.FAISS;
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        int dimension = 16;

        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            knnEngine,
            spaceType,
            new MethodComponentContext(METHOD_HNSW, ImmutableMap.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 512))
        );
        knnMethodContext.getMethodComponentContext().setIndexVersion(Version.CURRENT);

        String parameterString = XContentFactory.jsonBuilder().map(knnEngine.getMethodAsMap(knnMethodContext)).toString();

        FieldInfo[] fieldInfoArray = new FieldInfo[] {
            KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName)
                .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
                .addAttribute(KNNConstants.KNN_ENGINE, knnEngine.getName())
                .addAttribute(KNNConstants.SPACE_TYPE, spaceType.getValue())
                .addAttribute(KNNConstants.PARAMETERS, parameterString)
                .build() };

        FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        SegmentWriteState state = new SegmentWriteState(null, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        long initialRefreshOperations = KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue();
        long initialMergeOperations = KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue();

        // Add documents to the field
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(null, state);
        RandomVectorDocValuesProducer randomVectorDocValuesProducer = new RandomVectorDocValuesProducer(docsInSegment, dimension);
        knn80DocValuesConsumer.addKNNBinaryField(fieldInfoArray[0], randomVectorDocValuesProducer, true, true);

        // The document should be created in the correct location
        String expectedFile = KNNCodecUtil.buildEngineFileName(segmentName, knnEngine.getVersion(), fieldName, knnEngine.getExtension());
        assertFileInCorrectLocation(state, expectedFile);

        // The footer should be valid
        assertValidFooter(state.directory, expectedFile);

        // The document should be readable by faiss
        assertLoadableByEngine(HNSW_METHODPARAMETERS, state, expectedFile, knnEngine, spaceType, dimension);

        // The graph creation statistics should be updated
        assertEquals(1 + initialRefreshOperations, (long) KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        assertEquals(1 + initialMergeOperations, (long) KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());
    }

    public void testAddKNNBinaryField_whenFaissBinary_thenAdded() throws IOException {
        String segmentName = String.format("test_segment%s", randomAlphaOfLength(4));
        int docsInSegment = 100;
        String fieldName = String.format("test_field%s", randomAlphaOfLength(4));

        KNNEngine knnEngine = KNNEngine.FAISS;
        SpaceType spaceType = SpaceType.HAMMING;
        VectorDataType dataType = VectorDataType.BINARY;
        int dimension = 16;

        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            knnEngine,
            spaceType,
            new MethodComponentContext(METHOD_HNSW, ImmutableMap.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 512))
        );
        knnMethodContext.getMethodComponentContext().setIndexVersion(Version.CURRENT);

        String parameterString = XContentFactory.jsonBuilder().map(knnEngine.getMethodAsMap(knnMethodContext)).toString();

        FieldInfo[] fieldInfoArray = new FieldInfo[] {
            KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName)
                .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
                .addAttribute(KNNConstants.KNN_ENGINE, knnEngine.getName())
                .addAttribute(KNNConstants.SPACE_TYPE, spaceType.getValue())
                .addAttribute(VECTOR_DATA_TYPE_FIELD, dataType.getValue())
                .addAttribute(KNNConstants.PARAMETERS, parameterString)
                .build() };

        FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        SegmentWriteState state = new SegmentWriteState(null, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        long initialRefreshOperations = KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue();
        long initialMergeOperations = KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue();

        // Add documents to the field
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(null, state);
        RandomVectorDocValuesProducer randomVectorDocValuesProducer = new RandomVectorDocValuesProducer(docsInSegment, dimension);
        knn80DocValuesConsumer.addKNNBinaryField(fieldInfoArray[0], randomVectorDocValuesProducer, true, true);

        // The document should be created in the correct location
        String expectedFile = KNNCodecUtil.buildEngineFileName(segmentName, knnEngine.getVersion(), fieldName, knnEngine.getExtension());
        assertFileInCorrectLocation(state, expectedFile);

        // The footer should be valid
        assertValidFooter(state.directory, expectedFile);

        // The document should be readable by faiss
        assertBinaryIndexLoadableByEngine(state, expectedFile, knnEngine, spaceType, dimension, dataType);

        // The graph creation statistics should be updated
        assertEquals(1 + initialRefreshOperations, (long) KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        assertEquals(1 + initialMergeOperations, (long) KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());
    }

    public void testAddKNNBinaryField_fromModel_faiss() throws IOException, ExecutionException, InterruptedException {
        // Generate a trained faiss model
        KNNEngine knnEngine = KNNEngine.FAISS;
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        int dimension = 16;
        String modelId = "test-model-id";

        float[][] trainingData = getRandomVectors(200, dimension);
        long trainingPtr = JNIService.transferVectors(0, trainingData);

        Map<String, Object> parameters = ImmutableMap.of(
            INDEX_DESCRIPTION_PARAMETER,
            "IVF4,Flat",
            KNNConstants.SPACE_TYPE,
            SpaceType.L2.getValue()
        );

        byte[] modelBytes = JNIService.trainIndex(parameters, dimension, trainingPtr, knnEngine);
        Model model = new Model(
            new ModelMetadata(
                knnEngine,
                spaceType,
                dimension,
                ModelState.CREATED,
                "timestamp",
                "Empty description",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.FLOAT
            ),
            modelBytes,
            modelId
        );
        JNICommons.freeVectorData(trainingPtr);

        // Setup the model cache to return the correct model
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.get(modelId)).thenReturn(model);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getSettings()).thenReturn(Settings.EMPTY);

        ClusterSettings clusterSettings = new ClusterSettings(
            Settings.builder().put(MODEL_CACHE_SIZE_LIMIT_SETTING.getKey(), "10kb").build(),
            ImmutableSet.of(MODEL_CACHE_SIZE_LIMIT_SETTING)
        );

        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        ModelCache.initialize(modelDao, clusterService);

        // Build the segment and field info
        String segmentName = String.format("test_segment%s", randomAlphaOfLength(4));
        int docsInSegment = 100;
        String fieldName = String.format("test_field%s", randomAlphaOfLength(4));

        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        FieldInfo[] fieldInfoArray = new FieldInfo[] {
            KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName)
                .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
                .addAttribute(MODEL_ID, modelId)
                .build() };

        FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        SegmentWriteState state = new SegmentWriteState(null, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        long initialRefreshOperations = KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue();
        long initialMergeOperations = KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue();

        // Add documents to the field
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(null, state);
        RandomVectorDocValuesProducer randomVectorDocValuesProducer = new RandomVectorDocValuesProducer(docsInSegment, dimension);
        knn80DocValuesConsumer.addKNNBinaryField(fieldInfoArray[0], randomVectorDocValuesProducer, true, true);

        // The document should be created in the correct location
        String expectedFile = KNNCodecUtil.buildEngineFileName(segmentName, knnEngine.getVersion(), fieldName, knnEngine.getExtension());
        assertFileInCorrectLocation(state, expectedFile);

        // The footer should be valid
        assertValidFooter(state.directory, expectedFile);

        // The document should be readable by faiss
        assertLoadableByEngine(HNSW_METHODPARAMETERS, state, expectedFile, knnEngine, spaceType, dimension);

        // The graph creation statistics should be updated
        assertEquals(1 + initialRefreshOperations, (long) KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        assertEquals(1 + initialMergeOperations, (long) KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
        assertNotEquals(0, (long) KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());

    }

    public void testMerge_exception() throws IOException {
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(null, null);
        expectThrows(RuntimeException.class, () -> knn80DocValuesConsumer.merge(null));
    }

    public void testAddSortedSetField() throws IOException {
        // Verify that the delegate will be called
        DocValuesConsumer delegate = mock(DocValuesConsumer.class);
        doNothing().when(delegate).addSortedSetField(null, null);
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(delegate, null);
        knn80DocValuesConsumer.addSortedSetField(null, null);
        verify(delegate, times(1)).addSortedSetField(null, null);
    }

    public void testAddSortedNumericField() throws IOException {
        // Verify that the delegate will be called
        DocValuesConsumer delegate = mock(DocValuesConsumer.class);
        doNothing().when(delegate).addSortedNumericField(null, null);
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(delegate, null);
        knn80DocValuesConsumer.addSortedNumericField(null, null);
        verify(delegate, times(1)).addSortedNumericField(null, null);
    }

    public void testAddSortedField() throws IOException {
        // Verify that the delegate will be called
        DocValuesConsumer delegate = mock(DocValuesConsumer.class);
        doNothing().when(delegate).addSortedField(null, null);
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(delegate, null);
        knn80DocValuesConsumer.addSortedField(null, null);
        verify(delegate, times(1)).addSortedField(null, null);
    }

    public void testAddNumericField() throws IOException {
        // Verify that the delegate will be called
        DocValuesConsumer delegate = mock(DocValuesConsumer.class);
        doNothing().when(delegate).addNumericField(null, null);
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(delegate, null);
        knn80DocValuesConsumer.addNumericField(null, null);
        verify(delegate, times(1)).addNumericField(null, null);
    }

    public void testClose() throws IOException {
        // Verify that the delegate will be called
        DocValuesConsumer delegate = mock(DocValuesConsumer.class);
        doNothing().when(delegate).close();
        KNN80DocValuesConsumer knn80DocValuesConsumer = new KNN80DocValuesConsumer(delegate, null);
        knn80DocValuesConsumer.close();
        verify(delegate, times(1)).close();
    }

    public void testAddBinaryField_luceneEngine_noInvocations_addKNNBinary() throws IOException {
        var fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test-field")
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .build();
        DocValuesProducer docValuesProducer = mock(DocValuesProducer.class);

        var delegate = mock(DocValuesConsumer.class);
        doNothing().when(delegate).addBinaryField(fieldInfo, docValuesProducer);

        var knn80DocValuesConsumer = spy(new KNN80DocValuesConsumer(delegate, null));

        knn80DocValuesConsumer.addBinaryField(fieldInfo, docValuesProducer);

        verify(delegate, times(1)).addBinaryField(fieldInfo, docValuesProducer);
        verify(knn80DocValuesConsumer, never()).addKNNBinaryField(any(), any(), eq(false), eq(true));
    }
}
