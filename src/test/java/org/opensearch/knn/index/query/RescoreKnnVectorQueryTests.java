/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;

import static org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase.*;
import static org.mockito.Mockito.*;

public class RescoreKnnVectorQueryTests extends OpenSearchTestCase {

    public static final String FIELD_NAME = "vector-field";

    private List<float[]> generateRandomInput(int count, int dimension) {
        List<float[]> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            vectors.add(randomVector(dimension));
        }
        return vectors;
    }

    private static void addDocuments(List<float[]> vectors, Directory directory) throws IOException {
        try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
            for (float[] vector : vectors) {
                Document document = new Document();
                KnnFloatVectorField vectorField = new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN);
                document.add(vectorField);
                w.addDocument(document);
                w.commit();
            }
        }
    }

    private List<Float> calculateTopScores(List<float[]> vectors, float[] queryVector, int k) {
        List<Float> scores = new ArrayList<>();
        for (float[] vector : vectors) {
            scores.add(VectorSimilarityFunction.EUCLIDEAN.compare(queryVector, vector));
        }
        scores.sort(Collections.reverseOrder());
        return scores.subList(0, k);
    }

    @SneakyThrows
    public void testRescoreQuery() {
        int docsCount = 15;
        int dimension = 4;
        int finalK = 5;
        try (Directory directory = newDirectory()) {
            List<float[]> vectors = generateRandomInput(docsCount, dimension);
            addDocuments(vectors, directory);
            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(dimension);
                // Will use matchAll docs query to get all elements, but with score equal to 1, so that
                // after rescore we can compare actual scores.
                Query innerQuery = new MatchAllDocsQuery();
                try (
                    MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class);
                    MockedStatic<KNNSettings> mockedKnnSettings = mockStatic(KNNSettings.class)
                ) {
                    mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
                    mockedKnnSettings.when(() -> KNNSettings.isConcurrentExactSearchEnabled(any())).thenReturn(false);
                    mockedKnnSettings.when(() -> KNNSettings.getConcurrentExactSearchMaxPartitionCount(any())).thenReturn(0);
                    mockedKnnSettings.when(() -> KNNSettings.getConcurrentExactSearchMinDocumentCount(any())).thenReturn(1);
                    IndexSearcher searcher = newSearcher(reader, true, false, false);
                    RescoreKNNVectorQuery rescoreKnnVectorQuery = new RescoreKNNVectorQuery(
                        "test-index",
                        innerQuery,
                        FIELD_NAME,
                        finalK,
                        queryVector,
                        1
                    );
                    TopDocs rescoredDocs = searcher.search(rescoreKnnVectorQuery, finalK);
                    assertEquals(finalK, rescoredDocs.scoreDocs.length);
                    List<Float> actualScores = new ArrayList<>();
                    for (int i = 0; i < rescoredDocs.scoreDocs.length; i++) {
                        actualScores.add(rescoredDocs.scoreDocs[i].score);
                    }
                    assertEquals(calculateTopScores(vectors, queryVector, finalK), actualScores);
                }
            }
        }
    }

    public void testRescore_withMockCalls() throws IOException {
        IndexReader reader = createTestIndexReader();
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.getIndexReader()).thenReturn(reader);
        Query mockedInnerQuery = mock(Query.class);
        Query mockedRewrittenQuery = mock(Query.class);
        Weight mockedWeight = mock(Weight.class);
        TaskExecutor taskExecutor = mock(TaskExecutor.class);
        when(searcher.rewrite(mockedInnerQuery)).thenReturn(mockedRewrittenQuery);
        when(searcher.createWeight(mockedRewrittenQuery, ScoreMode.TOP_SCORES, 1.0f)).thenReturn(mockedWeight);
        when(searcher.getTaskExecutor()).thenReturn(taskExecutor);
        ExactSearcher mockedExactSearcher = mock(ExactSearcher.class);
        when(mockedWeight.scorer(any())).thenReturn(KNNScorer.emptyScorer());
        TopDocs exactSearchResults = generateRandomTopDocs(2);
        when(mockedExactSearcher.searchLeaf(any(), any())).thenReturn(exactSearchResults);
        when(taskExecutor.invokeAll(any())).thenAnswer(invocationOnMock -> {
            List<Callable<TopDocs>> callables = invocationOnMock.getArgument(0);
            List<TopDocs> results = new ArrayList<>();
            for (Callable<TopDocs> callable : callables) {
                results.add(callable.call());
            }
            return results;
        });
        try (
            MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class);
            MockedStatic<KNNSettings> mockedKnnSettings = mockStatic(KNNSettings.class)
        ) {
            mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
            mockedKnnSettings.when(() -> KNNSettings.isConcurrentExactSearchEnabled(any())).thenReturn(false);
            mockedKnnSettings.when(() -> KNNSettings.getConcurrentExactSearchMaxPartitionCount(any())).thenReturn(0);
            mockedKnnSettings.when(() -> KNNSettings.getConcurrentExactSearchMinDocumentCount(any())).thenReturn(1);
            RescoreKNNVectorQuery rescoreKnnVectorQuery = new RescoreKNNVectorQuery(
                "test-index",
                mockedInnerQuery,
                FIELD_NAME,
                10,
                new float[] { 1, 2 },
                1,
                mockedExactSearcher
            );
            Weight queryWeight = rescoreKnnVectorQuery.createWeight(searcher, ScoreMode.TOP_SCORES, 1.0f);
            assertNotNull(queryWeight);
            verify(searcher).rewrite(mockedInnerQuery);
            verify(searcher).createWeight(mockedRewrittenQuery, ScoreMode.TOP_SCORES, 1);
            verify(mockedExactSearcher).searchLeaf(any(), any());
            Query expectedQuery = QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), exactSearchResults);
            assertEquals(expectedQuery, queryWeight.getQuery());
        }

    }

    private TopDocs generateRandomTopDocs(int numDocs) {
        ScoreDoc[] scoreDocs = new ScoreDoc[numDocs];
        for (int i = 0; i < numDocs; i++) {
            scoreDocs[i] = new ScoreDoc(i, randomFloat());
        }
        return new TopDocs(new TotalHits(numDocs, TotalHits.Relation.EQUAL_TO), scoreDocs);
    }

    private IndexReader createTestIndexReader() throws IOException {
        ByteBuffersDirectory directory = new ByteBuffersDirectory();
        IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(new MockAnalyzer(random())));
        writer.addDocument(new Document());
        writer.close();
        return DirectoryReader.open(directory);
    }
}
