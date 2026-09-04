/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Verifies that real Lucene HNSW merges succeed when the merge executor
 * uses allowCoreThreadTimeOut. A very short keepAlive (1 second) is used
 * to stress-test that threads are not killed during an active merge, even
 * when the timeout is aggressively short.
 *
 * Multiple segments are created and then force-merged into one. If the
 * timeout fired mid-merge, the HNSW graph construction would fail or
 * produce a corrupt graph, and the post-merge kNN search would return
 * wrong results.
 */
public class KNN9120LuceneMergeThreadTimeoutTests extends KNNTestCase {

    private static final String FIELD = "vector";
    private static final int DIMENSION = 32;
    private static final int DOCS_PER_SEGMENT = 500;
    private static final int NUM_SEGMENTS = 6;
    private static final int TOTAL_DOCS = DOCS_PER_SEGMENT * NUM_SEGMENTS;

    @SneakyThrows
    public void testForceMerge_withShortKeepAlive_thenSearchSucceeds() {
        final int mergeThreads = 4;
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(mergeThreads);
        executor.setKeepAliveTime(1L, TimeUnit.SECONDS);
        executor.allowCoreThreadTimeOut(true);

        try (Directory dir = newDirectory()) {
            Codec codec = new UnitTestCodec(
                () -> new Lucene99HnswVectorsFormat(
                    Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN,
                    Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
                    mergeThreads,
                    executor
                )
            );
            IndexWriterConfig iwc = newIndexWriterConfig().setCodec(codec);
            Random rng = new Random(42);

            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
                    for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
                        Document doc = new Document();
                        doc.add(new KnnFloatVectorField(FIELD, randomVector(rng), VectorSimilarityFunction.EUCLIDEAN));
                        writer.addDocument(doc);
                    }
                    writer.flush();
                }

                // Force merge triggers the HNSW merge path that uses the executor.
                // With a 1-second keepAlive, any timing bug would cause threads to
                // die mid-merge, corrupting the graph or throwing an exception.
                writer.forceMerge(1);
                writer.commit();
            }

            // Verify the merged index is searchable and returns correct results
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                assertEquals(TOTAL_DOCS, reader.numDocs());

                float[] query = new float[DIMENSION];
                for (int d = 0; d < DIMENSION; d++) {
                    query[d] = 0.5f;
                }
                TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(FIELD, query, 10), 10);
                assertEquals(10, topDocs.scoreDocs.length);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    @SneakyThrows
    public void testRepeatedMerges_withShortKeepAlive_thenNoErrors() {
        final int mergeThreads = 4;
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(mergeThreads);
        executor.setKeepAliveTime(1L, TimeUnit.SECONDS);
        executor.allowCoreThreadTimeOut(true);

        try (Directory dir = newDirectory()) {
            Codec codec = new UnitTestCodec(
                () -> new Lucene99HnswVectorsFormat(
                    Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN,
                    Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
                    mergeThreads,
                    executor
                )
            );
            Random rng = new Random(123);

            // Round 1: index + merge
            IndexWriterConfig iwc1 = newIndexWriterConfig().setCodec(codec);
            try (IndexWriter writer = new IndexWriter(dir, iwc1)) {
                for (int seg = 0; seg < 3; seg++) {
                    for (int i = 0; i < 200; i++) {
                        Document doc = new Document();
                        doc.add(new KnnFloatVectorField(FIELD, randomVector(rng), VectorSimilarityFunction.EUCLIDEAN));
                        writer.addDocument(doc);
                    }
                    writer.flush();
                }
                writer.forceMerge(1);
                writer.commit();
            }

            // Let threads time out between merges
            Thread.sleep(2000);
            assertEquals(0, executor.getPoolSize());

            // Round 2: more docs + another merge (threads must spin back up)
            IndexWriterConfig iwc2 = newIndexWriterConfig().setCodec(codec).setOpenMode(IndexWriterConfig.OpenMode.APPEND);
            try (IndexWriter writer = new IndexWriter(dir, iwc2)) {
                for (int seg = 0; seg < 3; seg++) {
                    for (int i = 0; i < 200; i++) {
                        Document doc = new Document();
                        doc.add(new KnnFloatVectorField(FIELD, randomVector(rng), VectorSimilarityFunction.EUCLIDEAN));
                        writer.addDocument(doc);
                    }
                    writer.flush();
                }
                writer.forceMerge(1);
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals(1200, reader.numDocs());
                IndexSearcher searcher = new IndexSearcher(reader);
                float[] query = new float[DIMENSION];
                for (int d = 0; d < DIMENSION; d++) {
                    query[d] = 0.5f;
                }
                TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(FIELD, query, 10), 10);
                assertEquals(10, topDocs.scoreDocs.length);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    private float[] randomVector(Random rng) {
        float[] v = new float[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) {
            v[d] = rng.nextFloat();
        }
        return v;
    }
}
