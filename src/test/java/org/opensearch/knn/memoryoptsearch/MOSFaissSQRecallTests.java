/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

/**
 * End-to-end recall test for Faiss SQ 1-bit (32x compression) with INNER_PRODUCT.
 * Ingests real vectors via Faiss1040ScalarQuantizedKnnVectorsFormat (builds .vec, .veb, .faiss),
 * then searches via Faiss1040ScalarQuantizedKnnVectorsReader and measures recall against
 * brute-force ground truth.
 */
public class MOSFaissSQRecallTests extends KNNTestCase {

    private static final int NUM_VECTORS = 1000;
    private static final int DIMENSION = 768;
    private static final int TOP_K = 100;
    private static final String FIELD_NAME = "test_field";
    private static final String SEGMENT_NAME = "_0";

    private static final String PARAMETERS_JSON = "{"
        + "\"index_description\":\"BHNSW16,Flat\","
        + "\"spaceType\":\"innerproduct\","
        + "\"name\":\"hnsw\","
        + "\"data_type\":\"float\","
        + "\"parameters\":{"
        + "\"ef_search\":256,"
        + "\"ef_construction\":256,"
        + "\"m\":16,"
        + "\"encoder\":{\"name\":\"sq\",\"bits\":1}"
        + "}"
        + "}";

    @SneakyThrows
    public void testRecallWithInnerProduct() {
        final float[][] vectors = generateNormalizedVectors(NUM_VECTORS, DIMENSION, 42);
        final float[] query = generateNormalizedVectors(1, DIMENSION, 99)[0];

        // Compute brute-force ground truth (top-k by inner product, highest first)
        final Set<Integer> groundTruth = computeGroundTruth(vectors, query, TOP_K);

        try (Directory directory = newDirectory()) {
            // --- Ingest: build .vec, .veb, .faiss ---
            final FieldInfo fieldInfo = createFieldInfo();
            final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
            final SegmentInfo segInfo = createSegmentInfo(directory, SEGMENT_NAME, NUM_VECTORS);
            final SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                directory,
                segInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT,
                FIELD_NAME
            );

            final Faiss1040ScalarQuantizedKnnVectorsFormat format = new Faiss1040ScalarQuantizedKnnVectorsFormat();

            try (KnnVectorsWriter writer = format.fieldsWriter(writeState)) {
                KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
                for (int i = 0; i < NUM_VECTORS; i++) {
                    fw.addValue(i, vectors[i]);
                }
                writer.flush(NUM_VECTORS, null);
                writer.finish();
            }

            // Set the files on SegmentInfo so that SegmentInfo.files() doesn't throw.
            // The writer created .vec, .veb, .faiss and other Lucene files in the directory.
            final Set<String> writtenFiles = new HashSet<>();
            for (String f : directory.listAll()) {
                writtenFiles.add(f);
            }
            segInfo.setFiles(writtenFiles);

            // --- Search via Faiss1040ScalarQuantizedKnnVectorsReader ---
            final SegmentReadState readState = new SegmentReadState(
                directory,
                segInfo,
                fieldInfos,
                IOContext.DEFAULT,
                FIELD_NAME
            );

            try (KnnVectorsReader reader = format.fieldsReader(readState)) {
                final KnnCollector collector = new TopKnnCollector(TOP_K, Integer.MAX_VALUE);
                final AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(null, NUM_VECTORS);

                reader.search(FIELD_NAME, query, collector, acceptDocs);

                final TopDocs topDocs = collector.topDocs();
                final ScoreDoc[] scoreDocs = topDocs.scoreDocs;

                // Compute recall
                final Set<Integer> retrieved = new HashSet<>();
                for (ScoreDoc sd : scoreDocs) {
                    retrieved.add(sd.doc);
                }

                int hits = 0;
                for (int docId : groundTruth) {
                    if (retrieved.contains(docId)) {
                        hits++;
                    }
                }

                double recall = (double) hits / TOP_K;
                System.out.println("=== MOSFaissSQRecallTest ===");
                System.out.println("Vectors: " + NUM_VECTORS + ", Dimension: " + DIMENSION + ", Top-K: " + TOP_K);
                System.out.println("Retrieved: " + scoreDocs.length + " results");
                System.out.println("Recall@" + TOP_K + ": " + recall + " (" + hits + "/" + TOP_K + ")");
                System.out.println("============================");
            }
        }
    }


    // --- Helpers ---

    /**
     * Generate random vectors and L2-normalize them so inner product is meaningful.
     */
    private float[][] generateNormalizedVectors(int numVectors, int dimension, long seed) {
        final Random rng = new Random(seed);
        final float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            float norm = 0f;
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = rng.nextFloat() * 2 - 1;
                norm += vectors[i][j] * vectors[i][j];
            }
            norm = (float) Math.sqrt(norm);
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] /= norm;
            }
        }
        return vectors;
    }

    /**
     * Brute-force top-k by inner product (highest score = best match).
     */
    private Set<Integer> computeGroundTruth(float[][] vectors, float[] query, int k) {
        // Min-heap of size k: keeps the top-k highest inner products
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> Float.compare(
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(vectors[a[0]], query),
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(vectors[b[0]], query)
        ));

        for (int i = 0; i < vectors.length; i++) {
            if (minHeap.size() < k) {
                minHeap.add(new int[] { i });
            } else {
                float currentScore = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(vectors[i], query);
                float minScore = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(vectors[minHeap.peek()[0]], query);
                if (currentScore > minScore) {
                    minHeap.poll();
                    minHeap.add(new int[] { i });
                }
            }
        }

        Set<Integer> result = new HashSet<>();
        while (!minHeap.isEmpty()) {
            result.add(minHeap.poll()[0]);
        }
        return result;
    }

    private FieldInfo createFieldInfo() {
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
            Map.of(
                KNNConstants.PARAMETERS, PARAMETERS_JSON,
                KNNConstants.VECTOR_DATA_TYPE_FIELD, "float",
                KNNConstants.KNN_ENGINE, "faiss",
                KNNConstants.SQ_CONFIG, "bits=1",
                KNNConstants.SPACE_TYPE, "innerproduct",
                "knn_field", "true"
            ),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            false,
            false
        );
    }

    private SegmentInfo createSegmentInfo(org.apache.lucene.store.Directory directory, String segName, int maxDoc) {
        return new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            segName,
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null
        );
    }
}
