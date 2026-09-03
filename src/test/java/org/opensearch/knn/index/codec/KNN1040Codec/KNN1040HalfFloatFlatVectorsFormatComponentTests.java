/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableScorerTestUtils;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

/**
 * Component test for {@link KNN1040HalfFloatFlatVectorsFormat}: index a few FP16 float vector documents through a
 * segment written with this format, read the segment back, and validate the scorer wiring, end-to-end search scores
 * (which must match {@link VectorSimilarityFunction#EUCLIDEAN} on the FP16-rounded vectors), unsupported operations,
 * and merge behavior under an index sort.
 */
public class KNN1040HalfFloatFlatVectorsFormatComponentTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 8;
    private static final int NUM_DOCS = 10;

    /**
     * Index float vectors, read them back, and validate the returned scorer.
     *
     * <p>The type assertions confirm the wiring is
     * {@code PrefetchableFlatVectorScorer -> KNN1040HalfFloatVectorScorer -> NativeEngines990KnnVectorsScorer}.
     * The final behavioral assertion additionally exercises the query-time scoring entry point.
     */
    @SneakyThrows
    public void testIndexAndRead_whenHalfFloatFlatVectorsFormat_thenScorerIsPrefetchableHalfFloatScorer() {
        try (Directory dir = newDirectory()) {
            final float[][] vectors = indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = flatVectorsReaderFor(reader, FIELD_NAME);

                // ---- validate the scorer wired into the reader ----
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

                // ---- validate the RandomVectorScorer returned for a float query is the prefetchable variant ----
                final RandomVectorScorer randomVectorScorer = flatVectorsReader.getRandomVectorScorer(FIELD_NAME, vectors[0]);
                assertTrue(
                    "half-float random vector scorer must be the prefetchable variant but was " + randomVectorScorer.getClass().getName(),
                    randomVectorScorer instanceof PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer
                );
            }
        }
    }

    /**
     * Runs an actual kNN search over the indexed FP16 vectors and validates the returned scores against
     * {@link VectorSimilarityFunction#EUCLIDEAN} computed on the FP16-rounded vectors: every hit's score matches, and
     * scores are returned in descending order.
     */
    @SneakyThrows
    public void testSearch_whenHalfFloatFlatVectorsFormat_thenScoresMatchEuclidean() {
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

    @SneakyThrows
    public void testGetByteVectorValues_whenCalled_thenThrows() {
        try (Directory dir = newDirectory()) {
            indexFloatDocs(dir);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = flatVectorsReaderFor(reader, FIELD_NAME);
                expectThrows(UnsupportedOperationException.class, () -> flatVectorsReader.getByteVectorValues(FIELD_NAME));
            }
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_whenUnknownField_thenThrows() {
        try (Directory dir = newDirectory()) {
            indexFloatDocs(dir);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = flatVectorsReaderFor(reader, FIELD_NAME);
                IllegalArgumentException e = expectThrows(
                    IllegalArgumentException.class,
                    () -> flatVectorsReader.getFloatVectorValues("nonexistent")
                );
                assertTrue(e.getMessage().contains("nonexistent"));
            }
        }
    }

    /**
     * Merges segments under a descending index sort, so later segments land entirely before earlier ones, and
     * checks every doc still has the right vector.
     *
     * <p>A merge that appends {@code docMap.get(doc)} reader-by-reader would emit descending doc ids at the first
     * reader boundary, tripping {@code DocsWithFieldSet}'s strictly-increasing check.
     */
    @SneakyThrows
    public void testMerge_whenIndexSorted_thenPreservesDocToVectorMapping() {
        final int docsPerSegment = 5;
        final int numSegments = 3;
        final int totalDocs = docsPerSegment * numSegments;
        final String sortFieldName = "sort_key";
        final String idFieldName = "id";

        try (Directory dir = newDirectory()) {
            final Codec codec = new Lucene104Codec() {
                @Override
                public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return new KNN1040HalfFloatFlatVectorsFormat();
                }
            };

            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec)
                .setIndexSort(new Sort(new SortField(sortFieldName, SortField.Type.LONG, true)));

            final float[][] vectors = generateVectors(totalDocs);

            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (int i = 0; i < totalDocs; i++) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                    doc.add(new NumericDocValuesField(sortFieldName, i));
                    doc.add(new StoredField(idFieldName, i));
                    writer.addDocument(doc);
                    if ((i + 1) % docsPerSegment == 0) {
                        writer.commit();
                    }
                }
                writer.forceMerge(1);
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("force merge should leave a single segment", 1, reader.leaves().size());
                LeafReader leaf = reader.leaves().get(0).reader();

                FloatVectorValues values = leaf.getFloatVectorValues(FIELD_NAME);
                assertNotNull(values);
                assertEquals(totalDocs, values.size());

                int seen = 0;
                int previousId = Integer.MAX_VALUE;
                KnnVectorValues.DocIndexIterator iterator = values.iterator();
                for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                    int id = leaf.storedFields().document(doc).getField(idFieldName).numericValue().intValue();

                    // Descending sort on an ascending key: ids must come back in descending order, which is what
                    // proves the segments actually interleaved during the merge.
                    assertTrue("docs should be in descending id order, got " + id + " after " + previousId, id < previousId);
                    previousId = id;

                    float[] actual = values.vectorValue(iterator.index());
                    float[] expected = roundTripFp16(vectors[id]);
                    for (int d = 0; d < DIMENSION; d++) {
                        assertEquals("vector mismatch for id=" + id + " dim=" + d, expected[d], actual[d], 0.0f);
                    }
                    seen++;
                }
                assertEquals(totalDocs, seen);
            }
        }
    }

    /**
     * Navigates a reader to the per-field {@link FlatVectorsReader} backing the given field (leaf 0). This format is
     * registered directly as the per-field {@code KnnVectorsFormat} (no HNSW wrapper in between), so the per-field
     * reader already {@code is} the {@link FlatVectorsReader} - no inner-field unwrap needed.
     */
    private static FlatVectorsReader flatVectorsReaderFor(final DirectoryReader reader, final String field) throws Exception {
        final LeafReader leafReader = reader.leaves().get(0).reader();
        final SegmentReader segmentReader = Lucene.segmentReader(leafReader);

        final KnnVectorsReader perFieldReader = segmentReader.getVectorReader();
        if (perFieldReader instanceof PerFieldKnnVectorsFormat.FieldsReader == false) {
            throw new IllegalStateException("expected a PerFieldKnnVectorsFormat.FieldsReader but was " + perFieldReader);
        }
        final KnnVectorsReader fieldReader = ((PerFieldKnnVectorsFormat.FieldsReader) perFieldReader).getFieldReader(field);
        if (fieldReader == null) {
            throw new IllegalStateException("expected a per-field reader for " + field);
        }
        return (FlatVectorsReader) fieldReader;
    }

    private float[][] indexFloatDocs(final Directory dir) throws Exception {
        final Codec codec = new UnitTestCodec(KNN1040HalfFloatFlatVectorsFormat::new);
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
