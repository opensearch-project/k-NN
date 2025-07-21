/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.query.lucene.LuceneEngineKnnVectorQuery;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.indices.ModelDao;

import java.util.Map;
import java.util.Set;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class LuceneKNNWeightTests extends KNNTestCase {
    final String FIELD_NAME = "test_vector_field";
    final float[] QUERY_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
    final byte[] BYTE_QUERY_VECTOR = new byte[] { 1, 2, 3, 4 };
    final int K = 2;
    final String EXACT_SEARCH_SPACE_TYPE = "l2";
    final float BOOST = 1.0f;
    final String PER_FIELD_FORMAT_KEY = "Lucene99HnswVectorsFormat";

    public void testScorer() throws Exception {
        ModelDao modelDao = mock(ModelDao.class);
        LuceneKNNWeight.initialize(modelDao);

        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        SegmentReader segmentReader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(segmentReader);

        FieldInfos fieldInfos = mock(FieldInfos.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(segmentReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(eq(FIELD_NAME))).thenReturn(fieldInfo);
        when(fieldInfo.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY)).thenReturn(PER_FIELD_FORMAT_KEY);
        when(fieldInfo.getVectorSimilarityFunction()).thenReturn(VectorSimilarityFunction.EUCLIDEAN);
        when(fieldInfo.hasVectorValues()).thenReturn(true);
        when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);

        Query luceneQuery = new KnnFloatVectorQuery(FIELD_NAME, QUERY_VECTOR, K);
        LuceneEngineKnnVectorQuery knnVectorQuery = new LuceneEngineKnnVectorQuery(luceneQuery, EXACT_SEARCH_SPACE_TYPE);
        LuceneKNNWeight luceneKNNWeight = new LuceneKNNWeight(knnVectorQuery, BOOST);

        final FSDirectory directory = mock(FSDirectory.class);
        when(segmentReader.directory()).thenReturn(directory);
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "0",
            100,
            false,
            false,
            KNNCodecVersion.CURRENT_DEFAULT,
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(Set.of());
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(segmentReader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        try (MockedStatic<KNNVectorValuesFactory> factoryMock = mockStatic(KNNVectorValuesFactory.class)) {
            FloatVectorValues luceneVectorValues = mock(FloatVectorValues.class);
            when(segmentReader.getFloatVectorValues(eq(FIELD_NAME))).thenReturn(luceneVectorValues);
            KNNFloatVectorValues vectorValues = mock(KNNFloatVectorValues.class);
            factoryMock.when(() -> KNNVectorValuesFactory.getVectorValues(eq(VectorDataType.FLOAT), eq(luceneVectorValues)))
                .thenReturn(vectorValues);

            when(vectorValues.nextDoc()).thenReturn(0, 1, NO_MORE_DOCS);
            when(vectorValues.getVector()).thenReturn(QUERY_VECTOR);

            KNNScorer scorer = (KNNScorer) luceneKNNWeight.scorer(leafReaderContext);
            assertNotNull(scorer);
        }
    }

    public void testScorerWithByteVectors() throws Exception {
        ModelDao modelDao = mock(ModelDao.class);
        LuceneKNNWeight.initialize(modelDao);

        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        SegmentReader segmentReader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(segmentReader);

        FieldInfos fieldInfos = mock(FieldInfos.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(segmentReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(eq(FIELD_NAME))).thenReturn(fieldInfo);
        when(fieldInfo.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY)).thenReturn(PER_FIELD_FORMAT_KEY);
        when(fieldInfo.getVectorSimilarityFunction()).thenReturn(VectorSimilarityFunction.EUCLIDEAN);
        when(fieldInfo.hasVectorValues()).thenReturn(true);
        when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);

        Query luceneQuery = new KnnByteVectorQuery(FIELD_NAME, BYTE_QUERY_VECTOR, K);
        LuceneEngineKnnVectorQuery knnVectorQuery = new LuceneEngineKnnVectorQuery(luceneQuery, EXACT_SEARCH_SPACE_TYPE);
        LuceneKNNWeight luceneKNNWeight = new LuceneKNNWeight(knnVectorQuery, BOOST);

        final FSDirectory directory = mock(FSDirectory.class);
        when(segmentReader.directory()).thenReturn(directory);
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "0",
            100,
            false,
            false,
            KNNCodecVersion.CURRENT_DEFAULT,
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(Set.of());
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(segmentReader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        try (MockedStatic<KNNVectorValuesFactory> factoryMock = mockStatic(KNNVectorValuesFactory.class)) {
            ByteVectorValues luceneVectorValues = mock(ByteVectorValues.class);
            when(segmentReader.getByteVectorValues(eq(FIELD_NAME))).thenReturn(luceneVectorValues);
            KNNByteVectorValues vectorValues = mock(KNNByteVectorValues.class);
            factoryMock.when(() -> KNNVectorValuesFactory.getVectorValues(eq(VectorDataType.BYTE), eq(luceneVectorValues)))
                .thenReturn(vectorValues);

            when(vectorValues.nextDoc()).thenReturn(0, 1, NO_MORE_DOCS);
            when(vectorValues.getVector()).thenReturn(BYTE_QUERY_VECTOR);

            KNNScorer scorer = (KNNScorer) luceneKNNWeight.scorer(leafReaderContext);
            assertNotNull(scorer);
        }
    }

    public void testSearchLeaf() throws Exception {
        ModelDao modelDao = mock(ModelDao.class);
        LuceneKNNWeight.initialize(modelDao);

        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        SegmentReader segmentReader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(segmentReader);

        FieldInfos fieldInfos = mock(FieldInfos.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(segmentReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(eq(FIELD_NAME))).thenReturn(fieldInfo);
        when(fieldInfo.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY)).thenReturn(PER_FIELD_FORMAT_KEY);
        when(fieldInfo.getVectorSimilarityFunction()).thenReturn(VectorSimilarityFunction.EUCLIDEAN);
        when(fieldInfo.hasVectorValues()).thenReturn(true);
        when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);

        Query luceneQuery = new KnnFloatVectorQuery(FIELD_NAME, QUERY_VECTOR, K);
        LuceneEngineKnnVectorQuery knnVectorQuery = new LuceneEngineKnnVectorQuery(luceneQuery, EXACT_SEARCH_SPACE_TYPE);
        LuceneKNNWeight luceneKNNWeight = new LuceneKNNWeight(knnVectorQuery, BOOST);

        final FSDirectory directory = mock(FSDirectory.class);
        when(segmentReader.directory()).thenReturn(directory);
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "0",
            100,
            false,
            false,
            KNNCodecVersion.CURRENT_DEFAULT,
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(Set.of());
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(segmentReader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        try (MockedStatic<KNNVectorValuesFactory> factoryMock = mockStatic(KNNVectorValuesFactory.class)) {
            FloatVectorValues luceneVectorValues = mock(FloatVectorValues.class);
            when(segmentReader.getFloatVectorValues(eq(FIELD_NAME))).thenReturn(luceneVectorValues);
            KNNFloatVectorValues vectorValues = mock(KNNFloatVectorValues.class);
            factoryMock.when(() -> KNNVectorValuesFactory.getVectorValues(eq(VectorDataType.FLOAT), eq(luceneVectorValues)))
                .thenReturn(vectorValues);

            when(vectorValues.nextDoc()).thenReturn(0, 1, 2, NO_MORE_DOCS);
            when(vectorValues.getVector()).thenReturn(
                new float[] { 5.0f, 6.0f, 7.0f, 8.0f },
                new float[] { 1.1f, 2.1f, 3.1f, 4.1f },
                new float[] { 10.0f, 11.0f, 12.0f, 13.0f }
            );

            TopDocs topDocs = luceneKNNWeight.searchLeaf(leafReaderContext, K);

            assertNotNull(topDocs);
            assertEquals(K, topDocs.scoreDocs.length);

            assertEquals("First result should be doc 1", 1, topDocs.scoreDocs[0].doc);
            assertEquals("Second result should be doc 0", 0, topDocs.scoreDocs[1].doc);
        }
    }

    public void testExplain() throws Exception {
        ModelDao modelDao = mock(ModelDao.class);
        LuceneKNNWeight.initialize(modelDao);

        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        SegmentReader segmentReader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(segmentReader);

        FieldInfos fieldInfos = mock(FieldInfos.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(segmentReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(eq(FIELD_NAME))).thenReturn(fieldInfo);
        when(fieldInfo.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY)).thenReturn(PER_FIELD_FORMAT_KEY);
        when(fieldInfo.getVectorSimilarityFunction()).thenReturn(VectorSimilarityFunction.EUCLIDEAN);
        when(fieldInfo.hasVectorValues()).thenReturn(true);
        when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);

        Query luceneQuery = new KnnFloatVectorQuery(FIELD_NAME, QUERY_VECTOR, K);
        LuceneEngineKnnVectorQuery knnVectorQuery = new LuceneEngineKnnVectorQuery(luceneQuery, EXACT_SEARCH_SPACE_TYPE);
        LuceneKNNWeight luceneKNNWeight = new LuceneKNNWeight(knnVectorQuery, BOOST);

        final FSDirectory directory = mock(FSDirectory.class);
        when(segmentReader.directory()).thenReturn(directory);
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "0",
            100,
            false,
            false,
            KNNCodecVersion.CURRENT_DEFAULT,
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(Set.of());
        final SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);
        when(segmentReader.getSegmentInfo()).thenReturn(segmentCommitInfo);

        try (MockedStatic<KNNVectorValuesFactory> factoryMock = mockStatic(KNNVectorValuesFactory.class)) {
            FloatVectorValues luceneVectorValues = mock(FloatVectorValues.class);
            when(segmentReader.getFloatVectorValues(eq(FIELD_NAME))).thenReturn(luceneVectorValues);
            KNNFloatVectorValues vectorValues = mock(KNNFloatVectorValues.class);
            factoryMock.when(() -> KNNVectorValuesFactory.getVectorValues(eq(VectorDataType.FLOAT), eq(luceneVectorValues)))
                .thenReturn(vectorValues);

            when(vectorValues.nextDoc()).thenReturn(0, NO_MORE_DOCS);
            when(vectorValues.getVector()).thenReturn(new float[] { 1.1f, 2.1f, 3.1f, 4.1f });

            Explanation explanation = luceneKNNWeight.explain(leafReaderContext, 0);
            assertNotNull(explanation);
            assertTrue(explanation.getDescription().contains("KNN exact search"));
            Explanation details = explanation.getDetails()[0];
            assertTrue(details.getDescription().contains("l2"));
        }
    }
}
