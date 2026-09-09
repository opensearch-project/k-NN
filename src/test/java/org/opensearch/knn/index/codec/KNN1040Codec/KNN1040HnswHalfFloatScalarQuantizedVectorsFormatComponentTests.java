/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
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
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableScorerTestUtils;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

/**
 * Component test for {@link KNN1040HnswHalfFloatScalarQuantizedVectorsFormat}: index a few float vector documents
 * through a segment written with this format (HNSW graph over SQ 1-bit-quantized, FP16-sourced vectors), read the
 * segment back, and validate the scorer wiring and end-to-end search behavior. As with the flat SQ component test,
 * scores are not compared against an analytic similarity function - 1-bit scalar quantization is lossy, so only
 * descending order and non-NaN scores are asserted.
 */
public class KNN1040HnswHalfFloatScalarQuantizedVectorsFormatComponentTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_sq_vector";
    private static final int DIMENSION = 8;
    private static final int NUM_DOCS = 10;

    /**
     * Index vectors through the HNSW + SQ 1-bit format, read them back, and validate the returned scorer.
     *
     * <p>{@link KNN1040HnswHalfFloatScalarQuantizedVectorsFormat} delegates flat storage and scoring to
     * {@link KNN1040HalfFloatScalarQuantizedVectorsFormat}, so the random vector scorer handed out for graph
     * traversal must still be prefetchable, same as the bare flat SQ format.
     */
    @SneakyThrows
    public void testIndexAndRead_whenHnswHalfFloatScalarQuantizedVectorsFormat_thenScorerIsPrefetchable() {
        try (Directory dir = newDirectory()) {
            final float[][] vectors = indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = PrefetchableScorerTestUtils.flatVectorsReaderFor(reader, FIELD_NAME);

                final RandomVectorScorer scorer = flatVectorsReader.getRandomVectorScorer(FIELD_NAME, vectors[0]);
                assertTrue(
                    "random vector scorer must be the prefetchable variant but was " + scorer.getClass().getName(),
                    scorer instanceof PrefetchableRandomVectorScorer
                );
                assertEquals("maxOrd should match number of vectors", NUM_DOCS, scorer.maxOrd());
            }
        }
    }

    /**
     * Runs an actual kNN search over the indexed vectors (via the HNSW graph). Quantization is lossy, so this only
     * checks that every doc is returned, scores are non-NaN, and scores come back in descending order.
     */
    @SneakyThrows
    public void testSearch_whenHnswHalfFloatScalarQuantizedVectorsFormat_thenScoresDescendingAndNotNaN() {
        try (Directory dir = newDirectory()) {
            final float[][] vectors = indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final IndexSearcher searcher = new IndexSearcher(reader);
                final float[] query = vectors[2];

                final TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(FIELD_NAME, query, NUM_DOCS), NUM_DOCS);

                assertEquals("all indexed docs should be returned", NUM_DOCS, topDocs.scoreDocs.length);

                float previousScore = Float.MAX_VALUE;
                for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    assertFalse("score should not be NaN for doc " + scoreDoc.doc, Float.isNaN(scoreDoc.score));
                    assertTrue("scores must be in descending order", scoreDoc.score <= previousScore);
                    previousScore = scoreDoc.score;
                }
            }
        }
    }

    @SneakyThrows
    public void testSearch_whenHnswHalfFloatScalarQuantizedVectorsFormatWithCosine_thenScoresDescendingAndNotNaN() {
        try (Directory dir = newDirectory()) {
            final float[][] vectors = indexFloatDocs(dir, VectorSimilarityFunction.COSINE);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final IndexSearcher searcher = new IndexSearcher(reader);
                final float[] query = vectors[2];

                final TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(FIELD_NAME, query, NUM_DOCS), NUM_DOCS);

                assertEquals("all indexed docs should be returned", NUM_DOCS, topDocs.scoreDocs.length);

                float previousScore = Float.MAX_VALUE;
                for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    assertFalse("score should not be NaN for doc " + scoreDoc.doc, Float.isNaN(scoreDoc.score));
                    assertTrue("scores must be in descending order", scoreDoc.score <= previousScore);
                    previousScore = scoreDoc.score;
                }
            }
        }
    }

    private float[][] indexFloatDocs(final Directory dir) throws Exception {
        return indexFloatDocs(dir, VectorSimilarityFunction.EUCLIDEAN);
    }

    private float[][] indexFloatDocs(final Directory dir, final VectorSimilarityFunction similarity) throws Exception {
        final Codec codec = new UnitTestCodec(KNN1040HnswHalfFloatScalarQuantizedVectorsFormat::new);
        final IndexWriterConfig iwc = newIndexWriterConfig().setCodec(codec);
        final float[][] vectors = generateVectors(NUM_DOCS);
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], similarity));
                writer.addDocument(doc);
            }
            writer.forceMerge(1);
            writer.commit();
        }
        return vectors;
    }

    private float[][] generateVectors(int count) {
        float[][] vectors = new float[count][DIMENSION];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < DIMENSION; d++) {
                vectors[i][d] = (random().nextFloat() * 2 - 1) * 10;
            }
        }
        return vectors;
    }
}
