/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.jni.JNIService;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyFloat;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link MemOptimizedScalarQuantizedIndexBuildStrategy}.
 * <p>
 * Each test writes vectors through Lucene's binary quantized format to produce .vec and .veb files,
 * then invokes the SQ build strategy to construct a Faiss HNSW index from those files.
 * We verify the .faiss file is written successfully (non-zero size).
 * <p>
 * Dense case: doc_id == vector_ordinal (sequential IDs, no gaps)
 * Sparse case: doc_id != vector_ordinal (IDs with gaps, simulating deleted docs or nested fields)
 */
public class MemOptimizedScalarQuantizedIndexBuildStrategyTests extends KNNTestCase {

    // 4 -> lower dimension test
    // 56 -> non-multiple-of-8 quantized bytes (7 bytes), regression test for remainder loop bug
    // 128 -> test dimension that's multiple of 8
    // 333 -> test odd dimension
    private static final int[] DIMENSIONS = new int[] { 4, 56, 128, 333 };
    private static final int NUM_VECTORS = 1234;
    private static final String FIELD_NAME = "test_field";
    private static final String SEGMENT_NAME = "_0";

    @SneakyThrows
    public void testBuildDenseInnerProduct() {
        int[] docIds = sequentialDocIds(NUM_VECTORS);
        doBuildAndVerifyForMultipleDimensions(
            docIds,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            SpaceType.INNER_PRODUCT.getValue(),
            1
        );
    }

    @SneakyThrows
    public void testBuildSparseInnerProduct() {
        int[] docIds = sparseDocIds(NUM_VECTORS);
        doBuildAndVerifyForMultipleDimensions(
            docIds,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            SpaceType.INNER_PRODUCT.getValue(),
            1
        );
    }

    @SneakyThrows
    public void testBuildDenseL2() {
        int[] docIds = sequentialDocIds(NUM_VECTORS);
        doBuildAndVerifyForMultipleDimensions(docIds, VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getValue(), 1);
    }

    @SneakyThrows
    public void testBuildSparseL2() {
        int[] docIds = sparseDocIds(NUM_VECTORS);
        doBuildAndVerifyForMultipleDimensions(docIds, VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getValue(), 1);
    }

    @SneakyThrows
    public void testBuildDenseInnerProductMultiThreaded() {
        int[] docIds = sequentialDocIds(NUM_VECTORS);
        doBuildAndVerifyForMultipleDimensions(
            docIds,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            SpaceType.INNER_PRODUCT.getValue(),
            4
        );
    }

    @SneakyThrows
    public void testBuildSparseInnerProductMultiThreaded() {
        int[] docIds = sparseDocIds(NUM_VECTORS);
        doBuildAndVerifyForMultipleDimensions(
            docIds,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            SpaceType.INNER_PRODUCT.getValue(),
            4
        );
    }

    @SneakyThrows
    public void testBuildDenseL2MultiThreaded() {
        int[] docIds = sequentialDocIds(NUM_VECTORS);
        doBuildAndVerifyForMultipleDimensions(docIds, VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getValue(), 4);
    }

    @SneakyThrows
    public void testBuildSparseL2MultiThreaded() {
        int[] docIds = sparseDocIds(NUM_VECTORS);
        doBuildAndVerifyForMultipleDimensions(docIds, VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getValue(), 4);
    }

    @SneakyThrows
    private void doBuildAndVerifyForMultipleDimensions(
        int[] docIds,
        VectorSimilarityFunction similarityFunction,
        String spaceType,
        int indexThreadQty
    ) {
        for (int dimension : DIMENSIONS) {
            doBuildAndVerify(dimension, docIds, similarityFunction, spaceType, indexThreadQty);
        }
    }

    @SneakyThrows
    private void doBuildAndVerify(
        int dimension,
        int[] docIds,
        VectorSimilarityFunction similarityFunction,
        String spaceType,
        int indexThreadQty
    ) {
        final int maxDoc = docIds[docIds.length - 1] + 1;
        final float[][] vectors = generateRandomVectors(NUM_VECTORS, dimension);
        final byte[] segmentId = StringHelper.randomId();

        try (Directory directory = newDirectory()) {
            // Step 1: Build .vec + .veb files via Lucene102 binary quantized format
            final FieldInfo fieldInfo = createFieldInfo(similarityFunction, dimension);
            final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
            final SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                Version.LATEST,
                Version.LATEST,
                SEGMENT_NAME,
                maxDoc,
                false,
                false,
                null,
                Collections.emptyMap(),
                segmentId,
                Collections.emptyMap(),
                null
            );
            final SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                directory,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT,
                FIELD_NAME
            );

            final Lucene104ScalarQuantizedVectorsFormat sqFormat = new Lucene104ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
            final DocsWithFieldSet docsWithFieldSet;
            final Map<Integer, float[]> docIdToVector = new HashMap<>();
            try (FlatVectorsWriter flatWriter = sqFormat.fieldsWriter(writeState)) {
                final FlatFieldVectorsWriter fieldWriter = flatWriter.addField(fieldInfo);
                for (int i = 0; i < vectors.length; i++) {
                    fieldWriter.addValue(docIds[i], vectors[i]);
                    docIdToVector.put(docIds[i], vectors[i]);
                }
                docsWithFieldSet = fieldWriter.getDocsWithFieldSet();
                flatWriter.flush(maxDoc, null);
                flatWriter.finish();
            }

            // Step 2: Build the Faiss SQ HNSW index
            final Map<String, Object> parameters = buildIndexParameters(spaceType, indexThreadQty);
            final String faissFileName = SEGMENT_NAME + "_" + FIELD_NAME + ".faiss";

            // Open a reader to extract QuantizedByteVectorValues (same as the writer does)
            final SegmentReadState readState = new SegmentReadState(
                writeState.directory,
                writeState.segmentInfo,
                writeState.fieldInfos,
                writeState.context,
                writeState.segmentSuffix
            );

            try (
                final FlatVectorsReader flatVectorsReader = sqFormat.fieldsReader(readState);
                final IndexOutput indexOutput = directory.createOutput(faissFileName, IOContext.DEFAULT)
            ) {
                // Extract QuantizedByteVectorValues via reflection (same as the writer)
                final FloatVectorValues floatVectorValues = flatVectorsReader.getFloatVectorValues(FIELD_NAME);
                final java.lang.reflect.Field f = floatVectorValues.getClass().getDeclaredField("quantizedVectorValues");
                f.setAccessible(true);
                final QuantizedByteVectorValues quantizedValues = (QuantizedByteVectorValues) f.get(floatVectorValues);

                final IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(indexOutput);

                final BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                    .field(fieldInfo.getName())
                    .knnEngine(BuiltinKNNEngine.FAISS)
                    .indexOutputWithBuffer(indexOutputWithBuffer)
                    .vectorDataType(VectorDataType.FLOAT)
                    .indexParameters(parameters)
                    .knnVectorValuesSupplier(
                        () -> KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, docsWithFieldSet, docIdToVector)
                    )
                    .totalLiveDocs(NUM_VECTORS)
                    .segmentWriteState(writeState)
                    .quantizedByteVectorValues(quantizedValues)
                    .build();

                MemOptimizedScalarQuantizedIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);
            }

            // Step 3: Verify the .faiss file was written with non-zero size
            assertTrue("Faiss index file should have been written with non-zero size", directory.fileLength(faissFileName) > 0);
        }
    }

    @SneakyThrows
    public void testBuildAndWriteIndex_releasesIndexOnBuildFailure() {
        // Given: mock JNIService so initFaissSQIndex returns a fake address,
        // and passSQVectorsWithCorrectionFactors throws to simulate a Phase 1 failure.
        final long fakeIndexAddress = 42L;

        KNNFloatVectorValues knnVectorValues = mock(KNNFloatVectorValues.class);
        when(knnVectorValues.docId()).thenReturn(-1).thenReturn(0);
        when(knnVectorValues.nextDoc()).thenReturn(0);
        when(knnVectorValues.getVector()).thenReturn(new float[] { 1.0f, 2.0f });
        when(knnVectorValues.dimension()).thenReturn(2);

        QuantizedByteVectorValues quantizedValues = mock(QuantizedByteVectorValues.class);
        when(quantizedValues.vectorValue(0)).thenReturn(new byte[] { 0x01 });
        when(quantizedValues.getCentroidDP()).thenReturn(1.0f);
        when(quantizedValues.size()).thenReturn(1);
        when(quantizedValues.getCorrectiveTerms(0)).thenReturn(new OptimizedScalarQuantizer.QuantizationResult(0.0f, 1.0f, 0.0f, 0));

        IndexOutputWithBuffer indexOutputWithBuffer = mock(IndexOutputWithBuffer.class);

        Map<String, Object> params = new HashMap<>();
        params.put("index", "param");

        BuildIndexParams buildIndexParams = BuildIndexParams.builder()
            .indexOutputWithBuffer(indexOutputWithBuffer)
            .knnEngine(BuiltinKNNEngine.FAISS)
            .vectorDataType(VectorDataType.FLOAT)
            .indexParameters(params)
            .knnVectorValuesSupplier(() -> knnVectorValues)
            .totalLiveDocs(1)
            .quantizedByteVectorValues(quantizedValues)
            .build();

        try (MockedStatic<JNIService> mockedJNIService = Mockito.mockStatic(JNIService.class)) {
            mockedJNIService.when(
                () -> JNIService.initFaissSQIndex(anyInt(), anyInt(), anyMap(), anyFloat(), anyInt(), any(BuiltinKNNEngine.class))
            ).thenReturn(fakeIndexAddress);

            // Phase 1 throws
            mockedJNIService.when(
                () -> JNIService.passSQVectorsWithCorrectionFactors(anyLong(), any(byte[].class), anyInt(), any(BuiltinKNNEngine.class))
            ).thenThrow(new RuntimeException("Simulated Phase 1 failure"));

            // When
            RuntimeException thrown = expectThrows(
                RuntimeException.class,
                () -> MemOptimizedScalarQuantizedIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams)
            );

            // Then
            assertEquals("Simulated Phase 1 failure", thrown.getMessage());
            mockedJNIService.verify(() -> JNIService.releaseSQIndex(eq(fakeIndexAddress), eq(BuiltinKNNEngine.FAISS)));
            mockedJNIService.verify(
                () -> JNIService.writeIndex(any(), anyLong(), any(BuiltinKNNEngine.class), anyMap(), eq(true)),
                Mockito.never()
            );
        }
    }

    private static FieldInfo createFieldInfo(VectorSimilarityFunction similarityFunction, int dimension) {
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Collections.emptyMap(),
            0,
            0,
            0,
            dimension,
            VectorEncoding.FLOAT32,
            similarityFunction,
            false,
            false
        );
    }

    private static Map<String, Object> buildIndexParameters(String spaceType, int indexThreadQty) {
        Map<String, Object> params = new HashMap<>();
        params.put("name", "hnsw");
        params.put("data_type", "binary");
        params.put("index_description", "BHNSW16,Flat");
        params.put("spaceType", spaceType);

        Map<String, Object> subParams = new HashMap<>();
        subParams.put("ef_search", 256);
        subParams.put("ef_construction", 256);
        subParams.put("m", 16);
        subParams.put("encoder", Collections.emptyMap());
        subParams.put("indexThreadQty", indexThreadQty);
        params.put("parameters", subParams);

        return params;
    }

    private static float[][] generateRandomVectors(int numVectors, int dimension) {
        Random random = new Random(42);
        float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }

    private static int[] sequentialDocIds(int count) {
        int[] ids = new int[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i;
        }
        return ids;
    }

    private static int[] sparseDocIds(int count) {
        int[] ids = new int[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i * 3 + 1;
        }
        return ids;
    }
}
