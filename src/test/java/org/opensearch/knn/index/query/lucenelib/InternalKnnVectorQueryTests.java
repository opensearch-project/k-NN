/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class InternalKnnVectorQueryTests extends TestCase {
    private Directory directory;
    private IndexWriter indexWriter;
    private IndexSearcher indexSearcher;
    private DirectoryReader reader;
    private static final String FLOAT_FIELD = "float_vector_field";
    private static final String BYTE_FIELD = "byte_vector_field";
    private static final String TEST_EXACT_SEARCH_SPACE_TYPE = "l2";
    private static final int TEST_K = 2;
    private static final int DIMENSION = 4;
    private static ModelDao modelDao;
    private static ExactSearcher exactSearcher;

    public void setUp() throws Exception {
        super.setUp();
        Directory directory = new ByteBuffersDirectory();
        IndexWriterConfig config = new IndexWriterConfig();
        indexWriter = new IndexWriter(directory, config);

        for (int i = 0; i < 5; i++) {
            Document doc = new Document();

            // Add float vector
            float[] floatVector = new float[DIMENSION];
            for (int j = 0; j < DIMENSION; j++) {
                floatVector[j] = i + j;
            }
            doc.add(new KnnFloatVectorField(FLOAT_FIELD, floatVector));

            // Add byte vector
            byte[] byteVector = new byte[DIMENSION];
            for (int j = 0; j < DIMENSION; j++) {
                byteVector[j] = (byte) (i + j);
            }
            doc.add(new KnnByteVectorField(BYTE_FIELD, byteVector));

            indexWriter.addDocument(doc);
        }
        indexWriter.commit();
        reader = DirectoryReader.open(indexWriter);
        modelDao = mock(ModelDao.class);
        exactSearcher = mock(ExactSearcher.class);
        InternalKnnFloatVectorQuery.initialize(modelDao);
        InternalKnnByteVectorQuery.initialize(modelDao);
    }

    public void testFloatVectorSearchLeaf() throws Exception {
        float[] floatVector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        InternalKnnFloatVectorQuery query = new InternalKnnFloatVectorQuery(
            FLOAT_FIELD,
            floatVector,
            TEST_K,
            null,
            TEST_EXACT_SEARCH_SPACE_TYPE
        );
        TopDocs mockResults = new TopDocs(
            new TotalHits(2, TotalHits.Relation.EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(0, 1.0f), new ScoreDoc(1, 8.0f) }
        );

        when(exactSearcher.searchLeaf(any(LeafReaderContext.class), any(ExactSearcher.ExactSearcherContext.class))).thenReturn(mockResults);

        LeafReaderContext leafReaderContext = reader.leaves().get(0);
        TopDocs results = query.searchLeaf(leafReaderContext, TEST_K, null);

        assertNotNull(results);
        assertEquals(2, results.scoreDocs.length);
    }

    public void testByteVectorSearchLeaf() throws Exception {
        byte[] byteVector = new byte[] { 1, 2, 3, 4 };
        InternalKnnByteVectorQuery query = new InternalKnnByteVectorQuery(
            BYTE_FIELD,
            byteVector,
            TEST_K,
            null,
            TEST_EXACT_SEARCH_SPACE_TYPE
        );
        TopDocs mockResults = new TopDocs(
            new TotalHits(2, TotalHits.Relation.EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(0, 1.0f), new ScoreDoc(1, 8.0f) }
        );

        when(exactSearcher.searchLeaf(any(LeafReaderContext.class), any(ExactSearcher.ExactSearcherContext.class))).thenReturn(mockResults);

        LeafReaderContext leafReaderContext = reader.leaves().get(0);
        TopDocs results = query.searchLeaf(leafReaderContext, TEST_K, null);

        assertNotNull(results);
        assertEquals(2, results.scoreDocs.length);
    }
}
