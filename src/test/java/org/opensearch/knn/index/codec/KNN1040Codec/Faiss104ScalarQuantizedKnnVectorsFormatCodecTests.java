/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.junit.After;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

import java.io.IOException;

/**
 * Codec-level integration tests for {@link Faiss104ScalarQuantizedKnnVectorsFormat}.
 * Tests the write path (index + flush) and flat vector readback through a real Lucene index.
 *
 * <p>Uses a high approximateThreshold to skip native HNSW graph building (which requires
 * the Faiss JNI library), focusing on the Lucene BBQ flat vector storage path.
 */
public class Faiss104ScalarQuantizedKnnVectorsFormatCodecTests extends KNNTestCase {

    // High threshold so graph building is skipped — we're testing the flat vector write/read path
    private static final int SKIP_GRAPH_THRESHOLD = Integer.MAX_VALUE;
    private static final String BBQ_FIELD = "bbq_vector";

    private Directory dir;
    private RandomIndexWriter indexWriter;

    @After
    public void tearDown() throws Exception {
        if (dir != null) {
            dir.close();
        }
        super.tearDown();
    }

    /**
     * Index a single float vector through the BBQ format, flush, and verify the vector
     * is readable back with correct values and dimensions.
     */
    @SneakyThrows
    public void testBBQFormat_whenSingleVectorIndexed_thenReadBackSucceeds() {
        setup();
        float[] vector = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };

        Document doc = new Document();
        doc.add(new KnnFloatVectorField(BBQ_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
        indexWriter.addDocument(doc);

        final IndexReader reader = indexWriter.getReader();
        indexWriter.flush();
        indexWriter.commit();
        indexWriter.close();

        IndexSearcher searcher = new IndexSearcher(reader);
        final LeafReader leafReader = searcher.getLeafContexts().get(0).reader();

        final FloatVectorValues floatVectorValues = leafReader.getFloatVectorValues(BBQ_FIELD);
        KnnVectorValues.DocIndexIterator iterator = floatVectorValues.iterator();
        iterator.nextDoc();
        assertArrayEquals(vector, floatVectorValues.vectorValue(iterator.index()), 0.0f);
        assertEquals(1, floatVectorValues.size());
        assertEquals(8, floatVectorValues.dimension());

        reader.close();
    }

    /**
     * Index multiple vectors, flush, and verify all are readable with correct count.
     */
    @SneakyThrows
    public void testBBQFormat_whenMultipleVectorsIndexed_thenAllReadBack() {
        setup();
        float[] vector1 = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        float[] vector2 = { 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };

        Document doc1 = new Document();
        doc1.add(new KnnFloatVectorField(BBQ_FIELD, vector1, VectorSimilarityFunction.EUCLIDEAN));
        indexWriter.addDocument(doc1);

        Document doc2 = new Document();
        doc2.add(new KnnFloatVectorField(BBQ_FIELD, vector2, VectorSimilarityFunction.EUCLIDEAN));
        indexWriter.addDocument(doc2);

        final IndexReader reader = indexWriter.getReader();
        indexWriter.flush();
        indexWriter.commit();
        indexWriter.close();

        IndexSearcher searcher = new IndexSearcher(reader);
        final LeafReader leafReader = searcher.getLeafContexts().get(0).reader();

        final FloatVectorValues floatVectorValues = leafReader.getFloatVectorValues(BBQ_FIELD);
        assertEquals(2, floatVectorValues.size());
        assertEquals(8, floatVectorValues.dimension());

        KnnVectorValues.DocIndexIterator iterator = floatVectorValues.iterator();
        iterator.nextDoc();
        assertArrayEquals(vector1, floatVectorValues.vectorValue(iterator.index()), 0.0f);
        iterator.nextDoc();
        assertArrayEquals(vector2, floatVectorValues.vectorValue(iterator.index()), 0.0f);

        reader.close();
    }

    /**
     * Verify that the format name is correctly reported.
     */
    public void testBBQFormat_formatName_thenCorrect() {
        final Faiss104ScalarQuantizedKnnVectorsFormat format = new Faiss104ScalarQuantizedKnnVectorsFormat(SKIP_GRAPH_THRESHOLD);
        assertEquals("Faiss104ScalarQuantizedKnnVectorsFormat", format.getName());
    }

    private void setup() throws IOException {
        dir = newFSDirectory(createTempDir());
        // Disable index checking on close — BBQ codec search isn't fully wired up yet,
        // and Lucene's check-index would attempt searches that would fail.
        ((BaseDirectoryWrapper) dir).setCheckIndexOnClose(false);
        final Codec codec = new UnitTestCodec(() -> new Faiss104ScalarQuantizedKnnVectorsFormat(SKIP_GRAPH_THRESHOLD));
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(codec);
        iwc.setUseCompoundFile(false);
        iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        indexWriter = new RandomIndexWriter(random(), dir, iwc);
    }
}
