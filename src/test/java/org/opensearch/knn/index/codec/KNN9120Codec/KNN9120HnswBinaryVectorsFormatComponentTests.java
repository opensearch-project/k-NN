/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableScorerTestUtils;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

/**
 * Component test for {@link KNN9120HnswBinaryVectorsFormat}: index a few binary (byte) vector documents through a
 * segment written with this format, read the segment back, and validate both the scorer wiring and end-to-end search
 * scores (which must match {@link KNNVectorSimilarityFunction#HAMMING}).
 */
public class KNN9120HnswBinaryVectorsFormatComponentTests extends KNNTestCase {

    private static final String BINARY_VECTOR_FIELD = "binary_field";
    private static final int BYTES_PER_VECTOR = 8;
    private static final int NUM_DOCS = 4;

    /**
     * Index binary vectors, read them back, and validate the returned scorer.
     *
     * <p>The type assertions confirm the wiring is {@code PrefetchableFlatVectorScorer -> KNN9120BinaryVectorScorer}.
     * The final behavioral assertion additionally exercises the query-time scoring entry point.
     */
    @SneakyThrows
    public void testIndexAndRead_whenBinaryVectorsFormat_thenScorerIsPrefetchableBinaryScorer() {
        try (Directory dir = newDirectory()) {
            indexBinaryDocs(dir);

            // ---- read the segment back and reach the per-field flat vectors reader ----
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = PrefetchableScorerTestUtils.flatVectorsReaderFor(reader, BINARY_VECTOR_FIELD);

                // ---- validate the scorer wired into the reader ----
                final FlatVectorsScorer scorer = flatVectorsReader.getFlatVectorScorer(BINARY_VECTOR_FIELD);
                assertTrue(
                    "flat vectors scorer must be prefetchable but was " + scorer.getClass().getName(),
                    scorer instanceof PrefetchableFlatVectorScorer
                );
                final FlatVectorsScorer delegate = PrefetchableScorerTestUtils.getDelegate((PrefetchableFlatVectorScorer) scorer);
                assertTrue(
                    "prefetchable scorer must delegate to the binary scorer but was " + delegate.getClass().getName(),
                    delegate instanceof KNN9120BinaryVectorScorer
                );

                // ---- validate the RandomVectorScorer returned for a binary query is the prefetchable variant ----
                // The query path returns a fixed-target scorer (an AbstractRandomVectorScorer), so the prefetch
                // wrapper applies just like it does on the float path.
                final RandomVectorScorer randomVectorScorer = flatVectorsReader.getRandomVectorScorer(BINARY_VECTOR_FIELD, binaryVector(0));
                assertTrue(
                    "binary random vector scorer must be the prefetchable variant but was " + randomVectorScorer.getClass().getName(),
                    randomVectorScorer instanceof PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer
                );
            }
        }
    }

    /**
     * Runs an actual kNN search over the indexed binary vectors and validates the returned scores against
     * {@link KNNVectorSimilarityFunction#HAMMING}: the exact-match document ranks first with score {@code 1.0}, every
     * hit's score equals the Hamming similarity for that document, and scores are returned in descending order.
     */
    @SneakyThrows
    public void testSearch_whenBinaryVectorsFormat_thenScoresMatchHamming() {
        try (Directory dir = newDirectory()) {
            indexBinaryDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final IndexSearcher searcher = new IndexSearcher(reader);
                final int queryDoc = 2;
                final byte[] query = binaryVector(queryDoc);

                final TopDocs topDocs = searcher.search(new KnnByteVectorQuery(BINARY_VECTOR_FIELD, query, NUM_DOCS), NUM_DOCS);

                assertEquals("all indexed docs should be returned", NUM_DOCS, topDocs.scoreDocs.length);

                // Exact match must rank first with the maximum Hamming score (distance 0 -> 1 / (1 + 0) = 1.0).
                assertEquals("exact-match doc must rank first", queryDoc, topDocs.scoreDocs[0].doc);
                assertEquals("exact-match score must be 1.0", 1.0f, topDocs.scoreDocs[0].score, 1e-6f);

                // Every hit's score must equal the Hamming similarity for that doc, and scores must be descending.
                // docIds map 1:1 to insertion order here (single segment, sequential adds, force-merged, no deletes).
                float previousScore = Float.MAX_VALUE;
                for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    final float expected = KNNVectorSimilarityFunction.HAMMING.compare(query, binaryVector(scoreDoc.doc));
                    assertEquals("score mismatch for doc " + scoreDoc.doc, expected, scoreDoc.score, 1e-6f);
                    assertTrue("scores must be in descending order", scoreDoc.score <= previousScore);
                    previousScore = scoreDoc.score;
                }
            }
        }
    }

    private void indexBinaryDocs(final Directory dir) throws Exception {
        final Codec codec = new UnitTestCodec(KNN9120HnswBinaryVectorsFormat::new);
        final IndexWriterConfig iwc = newIndexWriterConfig().setCodec(codec);
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new KnnByteVectorField(BINARY_VECTOR_FIELD, binaryVector(i), VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.forceMerge(1);
            writer.commit();
        }
    }

    private static byte[] binaryVector(final int seed) {
        final byte[] vector = new byte[BYTES_PER_VECTOR];
        for (int i = 0; i < BYTES_PER_VECTOR; i++) {
            vector[i] = (byte) (seed * 31 + i);
        }
        return vector;
    }
}
