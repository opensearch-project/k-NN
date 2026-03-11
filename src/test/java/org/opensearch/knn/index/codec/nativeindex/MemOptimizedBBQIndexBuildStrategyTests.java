/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

/**
 * Tests for {@link MemOptimizedBBQIndexBuildStrategy}.
 * <p>
 * Each test writes vectors through Lucene's binary quantized format to produce .vec and .veb files,
 * then invokes the BBQ build strategy to construct a Faiss HNSW index from those files.
 * We verify the .faiss file is written successfully (non-zero size).
 * <p>
 * Dense case: doc_id == vector_ordinal (sequential IDs, no gaps)
 * Sparse case: doc_id != vector_ordinal (IDs with gaps, simulating deleted docs or nested fields)
 */
public class MemOptimizedBBQIndexBuildStrategyTests extends KNNTestCase {

    private static final int DIMENSION = 128;
    private static final int NUM_VECTORS = 70000;
    private static final String FIELD_NAME = "test_field";
    private static final String SEGMENT_NAME = "_0";

    @SneakyThrows
    public void testBuildDenseInnerProduct() {
        int[] docIds = sequentialDocIds(NUM_VECTORS);
        doBuildAndVerify(docIds, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, SpaceType.INNER_PRODUCT.getValue(), 1);
    }

    @SneakyThrows
    public void testBuildSparseInnerProduct() {
        int[] docIds = sparseDocIds(NUM_VECTORS);
        doBuildAndVerify(docIds, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, SpaceType.INNER_PRODUCT.getValue(), 1);
    }

    @SneakyThrows
    public void testBuildDenseL2() {
        int[] docIds = sequentialDocIds(NUM_VECTORS);
        doBuildAndVerify(docIds, VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getValue(), 1);
    }

    @SneakyThrows
    public void testBuildSparseL2() {
        int[] docIds = sparseDocIds(NUM_VECTORS);
        doBuildAndVerify(docIds, VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getValue(), 1);
    }

    @SneakyThrows
    public void testBuildDenseInnerProductMultiThreaded() {
        int[] docIds = sequentialDocIds(NUM_VECTORS);
        doBuildAndVerify(docIds, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, SpaceType.INNER_PRODUCT.getValue(), 4);
    }

    @SneakyThrows
    public void testBuildSparseInnerProductMultiThreaded() {
        int[] docIds = sparseDocIds(NUM_VECTORS);
        doBuildAndVerify(docIds, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, SpaceType.INNER_PRODUCT.getValue(), 4);
    }

    @SneakyThrows
    public void testBuildDenseL2MultiThreaded() {
        int[] docIds = sequentialDocIds(NUM_VECTORS);
        doBuildAndVerify(docIds, VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getValue(), 4);
    }

    @SneakyThrows
    public void testBuildSparseL2MultiThreaded() {
        int[] docIds = sparseDocIds(NUM_VECTORS);
        doBuildAndVerify(docIds, VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getValue(), 4);
    }

    @SneakyThrows
    private void doBuildAndVerify(int[] docIds, VectorSimilarityFunction similarityFunction, String spaceType, int indexThreadQty) {
        final int maxDoc = docIds[docIds.length - 1] + 1;
        final float[][] vectors = generateRandomVectors(NUM_VECTORS, DIMENSION);
        final byte[] segmentId = StringHelper.randomId();

        try (Directory directory = newDirectory()) {
            // Step 1: Build .vec + .veb files via Lucene102 binary quantized format
            final FieldInfo fieldInfo = createFieldInfo(similarityFunction);
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

            final Lucene104ScalarQuantizedVectorsFormat bbqFormat = new Lucene104ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
            final DocsWithFieldSet docsWithFieldSet;
            final Map<Integer, float[]> docIdToVector = new HashMap<>();
            try (FlatVectorsWriter flatWriter = bbqFormat.fieldsWriter(writeState)) {
                final FlatFieldVectorsWriter fieldWriter = flatWriter.addField(fieldInfo);
                for (int i = 0; i < vectors.length; i++) {
                    fieldWriter.addValue(docIds[i], vectors[i]);
                    docIdToVector.put(docIds[i], vectors[i]);
                }
                docsWithFieldSet = fieldWriter.getDocsWithFieldSet();
                flatWriter.flush(maxDoc, null);
                flatWriter.finish();
            }

            // Step 2: Build the Faiss BBQ HNSW index
            // Create a doc-id-to-vector map for KNNVectorValues
            final Map<String, Object> parameters = buildIndexParameters(spaceType, indexThreadQty);
            final String faissFileName = SEGMENT_NAME + "_" + FIELD_NAME + ".faiss";

            try (final IndexOutput indexOutput = directory.createOutput(faissFileName, IOContext.DEFAULT)) {
                final IndexOutputWithBuffer indexOutputWithBuffer = new IndexOutputWithBuffer(indexOutput);

                final BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                    .fieldInfo(fieldInfo)
                    .knnEngine(KNNEngine.FAISS)
                    .indexOutputWithBuffer(indexOutputWithBuffer)
                    .vectorDataType(VectorDataType.FLOAT)
                    .parameters(parameters)
                    .knnVectorValuesSupplier(
                        () -> KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, docsWithFieldSet, docIdToVector)
                    )
                    .totalLiveDocs(NUM_VECTORS)
                    .segmentWriteState(writeState)
                    .build();

                MemOptimizedBBQIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);
            }

            // Step 3: Verify the .faiss file was written with non-zero size
            assertTrue("Faiss index file should have been written with non-zero size", directory.fileLength(faissFileName) > 0);
        }
    }

    private static FieldInfo createFieldInfo(VectorSimilarityFunction similarityFunction) {
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
            DIMENSION,
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
