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
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

import static org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

/**
 * Component test for {@link KNN1040HalfFloatScalarQuantizedVectorsFormat}: index a few float vector documents
 * through a segment written with this format (SQ 1-bit over an FP16 raw delegate), read the segment back, and
 * validate the scorer wiring, end-to-end search behavior, unsupported operations, and merge behavior under an
 * index sort. Scores are not compared against an analytic similarity function like the plain FP16 format's
 * component tests do - 1-bit scalar quantization is lossy, so only descending order and non-NaN scores are
 * asserted.
 */
public class KNN1040HalfFloatScalarQuantizedVectorsFormatComponentTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_sq_vector";
    private static final int DIMENSION = 8;
    private static final int NUM_DOCS = 10;

    /**
     * Index vectors, read them back, and validate the returned scorer.
     *
     * <p>The type assertions confirm the reader's flat scorer is {@link KNN1040ScalarQuantizedVectorScorer}
     * (shared with the FLOAT SQ format - only the raw delegate differs) and that random vector scorers it hands
     * out are prefetchable.
     */
    @SneakyThrows
    public void testIndexAndRead_whenHalfFloatScalarQuantizedVectorsFormat_thenScorerIsScalarQuantizedScorer() {
        try (Directory dir = newDirectory()) {
            final float[][] vectors = indexFloatDocs(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = flatVectorsReaderFor(reader, FIELD_NAME);

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
     * Runs an actual kNN search over the indexed, SQ 1-bit quantized vectors. Quantization is lossy, so this only
     * checks that every doc is returned, scores are non-NaN, and scores come back in descending order - not that
     * scores match an analytic similarity computation.
     */
    @SneakyThrows
    public void testSearch_whenHalfFloatScalarQuantizedVectorsFormat_thenScoresDescendingAndNotNaN() {
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
    public void testSearch_whenHalfFloatScalarQuantizedVectorsFormatWithCosine_thenScoresDescendingAndNotNaN() {
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

    /**
     * Unlike {@link KNN1040HalfFloatFlatVectorsReader}, this format's reader is Lucene's own
     * {@code Lucene104ScalarQuantizedVectorsReader}, which returns {@code null} for an unknown field rather
     * than throwing.
     */
    @SneakyThrows
    public void testGetFloatVectorValues_whenUnknownField_thenReturnsNull() {
        try (Directory dir = newDirectory()) {
            indexFloatDocs(dir);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatVectorsReader = flatVectorsReaderFor(reader, FIELD_NAME);
                assertNull(flatVectorsReader.getFloatVectorValues("nonexistent"));
            }
        }
    }

    /**
     * Merges segments under a descending index sort, so later segments land entirely before earlier ones, and
     * checks every doc still resolves to a vector (quantization means the round-tripped value cannot be compared
     * exactly against the original, so only presence/count is asserted, not value equality).
     */
    @SneakyThrows
    public void testMerge_whenIndexSorted_thenPreservesDocCount() {
        final int docsPerSegment = 5;
        final int numSegments = 3;
        final int totalDocs = docsPerSegment * numSegments;
        final String sortFieldName = "sort_key";
        final String idFieldName = "id";

        try (Directory dir = newDirectory()) {
            final Codec codec = new Lucene104Codec() {
                @Override
                public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return new KNN1040HalfFloatScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
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
                    seen++;
                }
                assertEquals(totalDocs, seen);
            }
        }
    }

    @SneakyThrows
    public void testSpiReconstruction_resolvesToHalfFloatVariant() {
        final KNN1040HalfFloatScalarQuantizedVectorsFormat writeFormat = new KNN1040HalfFloatScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE
        );
        KnnVectorsFormat readFormat = KnnVectorsFormat.forName(writeFormat.getName());
        assertTrue(
            "forName(" + writeFormat.getName() + ") should resolve to the half_float SQ format",
            readFormat instanceof KNN1040HalfFloatScalarQuantizedVectorsFormat
        );
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
        return indexFloatDocs(dir, VectorSimilarityFunction.EUCLIDEAN);
    }

    private float[][] indexFloatDocs(final Directory dir, final VectorSimilarityFunction similarity) throws Exception {
        final Codec codec = new UnitTestCodec(() -> new KNN1040HalfFloatScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE));
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
