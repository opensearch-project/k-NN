/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
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
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableScorerTestUtils;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

/**
 * Component test for {@link KNN1040HnswHalfFloatVectorsFormat}: index a few FP16 float vector documents through a
 * segment written with this format, read the segment back, and validate both the scorer wiring and end-to-end search
 * scores (which must match {@link VectorSimilarityFunction#EUCLIDEAN} on the FP16-rounded vectors).
 */
public class KNN1040HnswHalfFloatVectorsFormatComponentTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 8;
    private static final int NUM_DOCS = 10;

    /**
     * Index FP16 vectors through the HNSW format, read them back, and validate the returned scorer.
     *
     * <p>The type assertions confirm the wiring is
     * {@code PrefetchableFlatVectorScorer -> KNN1040HalfFloatVectorScorer -> NativeEngines990KnnVectorsScorer},
     * same as the underlying flat format, since {@link KNN1040HnswHalfFloatVectorsFormat} delegates flat storage
     * and scoring to {@link KNN1040HalfFloatFlatVectorsFormat}.
     */
    @SneakyThrows
    public void testIndexAndRead_whenHnswHalfFloatVectorsFormat_thenScorerIsPrefetchableHalfFloatScorer() {
        try (Directory dir = newDirectory()) {
            final float[][] vectors = indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = PrefetchableScorerTestUtils.flatVectorsReaderFor(reader, FIELD_NAME);

                final FlatVectorsScorer scorer = flatVectorsReader.getFlatVectorScorer(FIELD_NAME);
                assertTrue(
                    "flat vectors scorer must be prefetchable but was " + scorer.getClass().getName(),
                    scorer instanceof PrefetchableFlatVectorScorer
                );
                final FlatVectorsScorer delegate = PrefetchableScorerTestUtils.getDelegate((PrefetchableFlatVectorScorer) scorer);
                assertTrue(
                    "prefetchable scorer must delegate to the half-float scorer but was " + delegate.getClass().getName(),
                    delegate instanceof KNN1040HalfFloatVectorScorer
                );

                final RandomVectorScorer randomVectorScorer = flatVectorsReader.getRandomVectorScorer(FIELD_NAME, vectors[0]);
                assertTrue(
                    "half-float random vector scorer must be the prefetchable variant but was " + randomVectorScorer.getClass().getName(),
                    randomVectorScorer instanceof PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer
                );
            }
        }
    }

    /**
     * Runs an actual kNN search over the indexed FP16 vectors (via the HNSW graph) and validates the returned scores
     * against {@link VectorSimilarityFunction#EUCLIDEAN} computed on the FP16-rounded vectors.
     */
    @SneakyThrows
    public void testSearch_whenHnswHalfFloatVectorsFormat_thenScoresMatchEuclidean() {
        try (Directory dir = newDirectory()) {
            final float[][] vectors = indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final IndexSearcher searcher = new IndexSearcher(reader);
                final float[] query = vectors[2];

                final TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(FIELD_NAME, query, NUM_DOCS), NUM_DOCS);

                assertEquals("all indexed docs should be returned", NUM_DOCS, topDocs.scoreDocs.length);

                float previousScore = Float.MAX_VALUE;
                for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    final float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, roundTripFp16(vectors[scoreDoc.doc]));
                    assertEquals("score mismatch for doc " + scoreDoc.doc, expected, scoreDoc.score, 1e-3f);
                    assertTrue("scores must be in descending order", scoreDoc.score <= previousScore);
                    previousScore = scoreDoc.score;
                }
            }
        }
    }

    private float[][] indexFloatDocs(final Directory dir) throws Exception {
        final Codec codec = new UnitTestCodec(KNN1040HnswHalfFloatVectorsFormat::new);
        final IndexWriterConfig iwc = newIndexWriterConfig().setCodec(codec);
        final float[][] vectors = generateVectors(NUM_DOCS);
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
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

    private static float[] roundTripFp16(float[] vector) {
        byte[] bytes = new byte[vector.length * Short.BYTES];
        KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, bytes, vector.length);
        float[] rounded = new float[vector.length];
        KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.byteToFloatArray(bytes, rounded, vector.length, 0);
        return rounded;
    }
}
