/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.lucene;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.junit.After;
import org.junit.Before;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableScorerTestUtils;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

/**
 * Component test for the stock {@link Lucene99HnswVectorsFormat} (float / fp32 HNSW), mirroring the binary format
 * component test. It indexes a few float vector documents, reads the segment back, and validates the scorer wired
 * into the flat vectors reader, plus end-to-end search scores:
 *
 * <ul>
 *   <li>Without the patch, the reader returns the stock (non-prefetchable) scorer.</li>
 *   <li>After {@link Lucene99ScorerPatcher#installOnce()}, the reader returns a {@link PrefetchableFlatVectorScorer},
 *       the float {@link RandomVectorScorer} is the prefetchable variant, and a kNN search still produces scores that
 *       match {@link VectorSimilarityFunction#EUCLIDEAN} — confirming the prefetch wrapper preserves scores.</li>
 * </ul>
 *
 * <p>{@code installOnce()} mutates the process-wide scorer on the shared {@code Lucene99FlatVectorsFormat}, so each
 * test resets that global state around itself (via {@link PrefetchableScorerTestUtils}).
 */
public class Lucene99HnswVectorsFormatComponentTests extends KNNTestCase {

    private static final String FLOAT_VECTOR_FIELD = "float_field";
    private static final int DIMENSION = 4;
    private static final int NUM_DOCS = 4;

    private PrefetchableScorerTestUtils.SharedScorerState scorerState;

    @Before
    @SneakyThrows
    public void resolveSharedScorerAndReset() {
        scorerState = PrefetchableScorerTestUtils.resolveAndReset();
    }

    @After
    @SneakyThrows
    public void restoreGlobalState() {
        if (scorerState != null) {
            scorerState.restore();
        }
    }

    @SneakyThrows
    public void testIndexAndRead_whenStockFormat_thenScorerIsNotPrefetchable() {
        try (Directory dir = newDirectory()) {
            indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = PrefetchableScorerTestUtils.flatVectorsReaderFor(reader, FLOAT_VECTOR_FIELD);

                final FlatVectorsScorer scorer = flatVectorsReader.getFlatVectorScorer(FLOAT_VECTOR_FIELD);
                assertFalse(
                    "unpatched stock format must not return a prefetchable scorer but was " + scorer.getClass().getName(),
                    scorer instanceof PrefetchableFlatVectorScorer
                );

                final RandomVectorScorer randomVectorScorer = flatVectorsReader.getRandomVectorScorer(FLOAT_VECTOR_FIELD, queryVector());
                assertFalse(
                    "unpatched random vector scorer must not be the prefetchable variant",
                    randomVectorScorer instanceof PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer
                );
            }
        }
    }

    @SneakyThrows
    public void testIndexAndRead_afterInstallOnce_thenScorerIsPrefetchable() {
        Lucene99ScorerPatcher.installOnce();

        try (Directory dir = newDirectory()) {
            indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = PrefetchableScorerTestUtils.flatVectorsReaderFor(reader, FLOAT_VECTOR_FIELD);

                final FlatVectorsScorer scorer = flatVectorsReader.getFlatVectorScorer(FLOAT_VECTOR_FIELD);
                assertTrue(
                    "patched stock format must return a prefetchable scorer but was " + scorer.getClass().getName(),
                    scorer instanceof PrefetchableFlatVectorScorer
                );

                // Unlike the binary scorer, the stock float scorer returns an AbstractRandomVectorScorer, so the
                // prefetch wrapper applies cleanly on the float path.
                final RandomVectorScorer randomVectorScorer = flatVectorsReader.getRandomVectorScorer(FLOAT_VECTOR_FIELD, queryVector());
                assertTrue(
                    "patched random vector scorer must be the prefetchable variant but was " + randomVectorScorer.getClass().getName(),
                    randomVectorScorer instanceof PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer
                );
            }
        }
    }

    /**
     * Runs an actual kNN search over the indexed float vectors after installing the prefetch patch, and validates the
     * returned scores against {@link VectorSimilarityFunction#EUCLIDEAN}: the exact-match document ranks first with
     * score {@code 1.0}, every hit's score equals the Euclidean similarity for that document, and scores are returned
     * in descending order. This confirms the prefetch wrapper preserves scores end-to-end on the float path.
     */
    @SneakyThrows
    public void testSearch_afterInstallOnce_thenScoresMatchEuclidean() {
        Lucene99ScorerPatcher.installOnce();

        try (Directory dir = newDirectory()) {
            indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final IndexSearcher searcher = new IndexSearcher(reader);
                final int queryDoc = 2;
                final float[] query = floatVector(queryDoc);

                final TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(FLOAT_VECTOR_FIELD, query, NUM_DOCS), NUM_DOCS);

                assertEquals("all indexed docs should be returned", NUM_DOCS, topDocs.scoreDocs.length);

                // Exact match must rank first with the maximum Euclidean score (distance 0 -> 1 / (1 + 0) = 1.0).
                assertEquals("exact-match doc must rank first", queryDoc, topDocs.scoreDocs[0].doc);
                assertEquals("exact-match score must be 1.0", 1.0f, topDocs.scoreDocs[0].score, 1e-6f);

                // Every hit's score must equal the Euclidean similarity for that doc, and scores must be descending.
                // docIds map 1:1 to insertion order here (single segment, sequential adds, force-merged, no deletes).
                float previousScore = Float.MAX_VALUE;
                for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    final float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, floatVector(scoreDoc.doc));
                    assertEquals("score mismatch for doc " + scoreDoc.doc, expected, scoreDoc.score, 1e-6f);
                    assertTrue("scores must be in descending order", scoreDoc.score <= previousScore);
                    previousScore = scoreDoc.score;
                }
            }
        }
    }

    private void indexFloatDocs(final Directory dir) throws Exception {
        final Codec codec = new UnitTestCodec(Lucene99HnswVectorsFormat::new);
        final IndexWriterConfig iwc = newIndexWriterConfig().setCodec(codec);
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FLOAT_VECTOR_FIELD, floatVector(i), VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.forceMerge(1);
            writer.commit();
        }
    }

    private static float[] floatVector(final int seed) {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = (seed + 1) * 0.1f + i * 0.01f;
        }
        return vector;
    }

    private static float[] queryVector() {
        return floatVector(0);
    }
}
