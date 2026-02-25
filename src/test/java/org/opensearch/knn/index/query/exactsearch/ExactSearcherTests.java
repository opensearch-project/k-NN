/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.KNNRestTestCase.FIELD_NAME;
import static org.opensearch.knn.KNNRestTestCase.INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class ExactSearcherTests extends KNNTestCase {

    private static final String SEGMENT_NAME = "0";

    @SneakyThrows
    public void testExactSearch_whenSegmentHasNoVectorField_thenNoDocsReturned() {
        final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
        final KNNQuery query = KNNQuery.builder().field(FIELD_NAME).queryVector(queryVector).k(10).indexName(INDEX_NAME).build();

        final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
            ExactSearcher.ExactSearcherContext.builder().field(FIELD_NAME).floatQueryVector(queryVector);

        ExactSearcher exactSearcher = new ExactSearcher(null);
        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(query.getField())).thenReturn(null);
        TopDocs docs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
        Mockito.verify(fieldInfos).fieldInfo(query.getField());
        Mockito.verify(reader).getFieldInfos();
        Mockito.verify(leafReaderContext).reader();
        assertEquals(0, docs.scoreDocs.length);
    }

    @SneakyThrows
    public void testExactSearch_whenMatchedDocsAndVectorValuesHasDifferentDocs_thenSuccess() {
        final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
        final float[] vector = new float[] { 0.3f, 4.0f, 1.0f };
        final SpaceType spaceType = SpaceType.L2;
        final KNNQuery query = KNNQuery.builder().field(FIELD_NAME).queryVector(queryVector).k(10).indexName(INDEX_NAME).build();

        DocIdSetIterator matchedDocIdSetIterator = DocIdSetIterator.all(10);

        final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
            ExactSearcher.ExactSearcherContext.builder()
                .field(FIELD_NAME)
                .floatQueryVector(queryVector)
                .matchedDocsIterator(matchedDocIdSetIterator);
        try (MockedStatic<KNNVectorValuesFactory> vectorValuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            ExactSearcher exactSearcher = new ExactSearcher(null);
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);

            final FieldInfos fieldInfos = mock(FieldInfos.class);
            final FieldInfo fieldInfo = mock(FieldInfo.class);
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(spaceType.getValue());
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(query.getField())).thenReturn(fieldInfo);
            final KNNVectorValues knnFloatVectorValues = TestVectorValues.createKNNFloatVectorValues(List.of(vector));
            when(leafReaderContext.reader()).thenReturn(reader);
            vectorValuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader))
                .thenReturn(knnFloatVectorValues);

            TopDocs docs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
            Mockito.verify(fieldInfos).fieldInfo(query.getField());
            Mockito.verify(reader).getFieldInfos();
            Mockito.verify(leafReaderContext).reader();
            assertEquals(1, docs.scoreDocs.length);
            assertEquals(spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector), docs.scoreDocs[0].score, 1e-6f);
        }
    }

    @SneakyThrows
    public void testExactSearch_whenMatchedDocsIsNull_thenSuccess() {
        final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
        final float[] vector = new float[] { 0.3f, 4.0f, 1.0f };
        final SpaceType spaceType = SpaceType.L2;
        int k = 10;
        final KNNQuery query = KNNQuery.builder().field(FIELD_NAME).queryVector(queryVector).k(k).indexName(INDEX_NAME).build();

        final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
            ExactSearcher.ExactSearcherContext.builder().field(FIELD_NAME).floatQueryVector(queryVector).k(k);
        try (MockedStatic<KNNVectorValuesFactory> vectorValuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            ExactSearcher exactSearcher = new ExactSearcher(null);
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);

            final FieldInfos fieldInfos = mock(FieldInfos.class);
            final FieldInfo fieldInfo = mock(FieldInfo.class);
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(spaceType.getValue());
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(query.getField())).thenReturn(fieldInfo);
            final KNNVectorValues knnFloatVectorValues = TestVectorValues.createKNNFloatVectorValues(List.of(vector));
            when(leafReaderContext.reader()).thenReturn(reader);
            vectorValuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader))
                .thenReturn(knnFloatVectorValues);

            TopDocs docs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
            Mockito.verify(fieldInfos).fieldInfo(query.getField());
            Mockito.verify(reader).getFieldInfos();
            Mockito.verify(leafReaderContext).reader();
            assertEquals(1, docs.scoreDocs.length);
            assertEquals(spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector), docs.scoreDocs[0].score, 1e-6f);
        }
    }

    @SneakyThrows
    public void testRadialSearchExactSearch_whenSegmentHasNoVectorField_thenNoDocsReturned() {
        final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
        KNNQuery.Context context = new KNNQuery.Context(10);
        final KNNQuery query = KNNQuery.builder()
            .field(FIELD_NAME)
            .queryVector(queryVector)
            .context(context)
            .radius(1.0f)
            .indexName(INDEX_NAME)
            .build();

        final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
            ExactSearcher.ExactSearcherContext.builder().field(FIELD_NAME).floatQueryVector(queryVector);

        ExactSearcher exactSearcher = new ExactSearcher(null);
        final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        final SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(query.getField())).thenReturn(null);
        TopDocs docs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
        Mockito.verify(fieldInfos).fieldInfo(query.getField());
        Mockito.verify(reader).getFieldInfos();
        Mockito.verify(leafReaderContext).reader();
        assertEquals(0, docs.scoreDocs.length);
    }

    @SneakyThrows
    public void testRadialSearch_whenNoEngineFiles_thenSuccess() {
        doTestRadialSearch_whenNoEngineFiles_thenSuccess(false);
        doTestRadialSearch_whenNoEngineFiles_thenSuccess(true);
    }

    @SneakyThrows
    public void testExactSearch_whenQuantizationWithoutADC_thenSuccess() {
        try (
            MockedStatic<KNNVectorValuesFactory> valuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<org.opensearch.knn.index.query.SegmentLevelQuantizationInfo> quantizationInfoMockedStatic = Mockito.mockStatic(
                org.opensearch.knn.index.query.SegmentLevelQuantizationInfo.class
            );
            MockedStatic<org.opensearch.knn.index.query.SegmentLevelQuantizationUtil> quantizationUtilMockedStatic = Mockito.mockStatic(
                org.opensearch.knn.index.query.SegmentLevelQuantizationUtil.class
            )
        ) {
            final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
            final byte[] quantizedQueryVector = new byte[] { 1, 0, 1 };
            final List<float[]> floatVectors = List.of(
                new float[] { 1, 2, 3 },
                new float[] { 4, 5, 6 },
                new float[] { 2, 3, 4 },
                new float[] { 7, 8, 9 },
                new float[] { 0, 1, 2 }
            );
            final List<byte[]> quantizedDataVectors = List.of(
                new byte[] { 1, 0, 1 },
                new byte[] { 0, 1, 0 },
                new byte[] { 1, 1, 0 },
                new byte[] { 0, 0, 1 },
                new byte[] { 1, 1, 1 }
            );
            final SpaceType spaceType = SpaceType.L2;
            final int k = 10;

            final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                .field(FIELD_NAME)
                .floatQueryVector(queryVector)
                .useQuantizedVectorsForSearch(true)
                .k(k)
                .build();

            ExactSearcher exactSearcher = new ExactSearcher(null);
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);

            final FieldInfos fieldInfos = mock(FieldInfos.class);
            final FieldInfo fieldInfo = mock(FieldInfo.class);
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(spaceType.getValue());
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(FIELD_NAME)).thenReturn(fieldInfo);

            final org.opensearch.knn.index.query.SegmentLevelQuantizationInfo quantizationInfo = mock(
                org.opensearch.knn.index.query.SegmentLevelQuantizationInfo.class
            );
            quantizationInfoMockedStatic.when(
                () -> org.opensearch.knn.index.query.SegmentLevelQuantizationInfo.build(reader, fieldInfo, FIELD_NAME)
            ).thenReturn(quantizationInfo);
            quantizationUtilMockedStatic.when(
                () -> org.opensearch.knn.index.query.SegmentLevelQuantizationUtil.isAdcEnabled(quantizationInfo)
            ).thenReturn(false);
            quantizationUtilMockedStatic.when(
                () -> org.opensearch.knn.index.query.SegmentLevelQuantizationUtil.quantizeVector(queryVector, quantizationInfo)
            ).thenReturn(quantizedQueryVector);

            final KNNVectorValues knnFloatVectorValues = TestVectorValues.createKNNFloatVectorValues(floatVectors);
            final KNNVectorValues knnByteVectorValues = TestVectorValues.createKNNBinaryVectorValues(quantizedDataVectors);
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader))
                .thenReturn(knnFloatVectorValues);
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader, true))
                .thenReturn(knnByteVectorValues);

            TopDocs docs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContext);
            assertEquals(quantizedDataVectors.size(), docs.scoreDocs.length);

            final List<Float> expectedScores = quantizedDataVectors.stream()
                .map(quantizedVector -> SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(quantizedQueryVector, quantizedVector))
                .sorted(Comparator.reverseOrder())
                .toList();
            final List<Float> actualScores = Arrays.stream(docs.scoreDocs).map(scoreDoc -> scoreDoc.score).toList();
            assertEquals(expectedScores, actualScores);
        }
    }

    @SneakyThrows
    public void testExactSearch_whenQuantizationWithADC_thenSuccess() {
        try (
            MockedStatic<KNNVectorValuesFactory> valuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<org.opensearch.knn.index.query.SegmentLevelQuantizationInfo> quantizationInfoMockedStatic = Mockito.mockStatic(
                org.opensearch.knn.index.query.SegmentLevelQuantizationInfo.class
            );
            MockedStatic<org.opensearch.knn.index.query.SegmentLevelQuantizationUtil> quantizationUtilMockedStatic = Mockito.mockStatic(
                org.opensearch.knn.index.query.SegmentLevelQuantizationUtil.class
            )
        ) {
            final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
            final List<float[]> floatVectors = List.of(
                new float[] { 1, 2, 3 },
                new float[] { 4, 5, 6 },
                new float[] { 2, 3, 4 },
                new float[] { 7, 8, 9 },
                new float[] { 0, 1, 2 }
            );
            final List<byte[]> quantizedDataVectors = List.of(
                new byte[] { 1, 0, 1 },
                new byte[] { 0, 1, 0 },
                new byte[] { 1, 1, 0 },
                new byte[] { 0, 0, 1 },
                new byte[] { 1, 1, 1 }
            );
            final SpaceType spaceType = SpaceType.L2;
            final int k = 10;

            final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                .field(FIELD_NAME)
                .floatQueryVector(queryVector)
                .useQuantizedVectorsForSearch(true)
                .k(k)
                .build();

            ExactSearcher exactSearcher = new ExactSearcher(null);
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);

            final FieldInfos fieldInfos = mock(FieldInfos.class);
            final FieldInfo fieldInfo = mock(FieldInfo.class);
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(spaceType.getValue());
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(FIELD_NAME)).thenReturn(fieldInfo);

            final org.opensearch.knn.index.query.SegmentLevelQuantizationInfo quantizationInfo = mock(
                org.opensearch.knn.index.query.SegmentLevelQuantizationInfo.class
            );
            quantizationInfoMockedStatic.when(
                () -> org.opensearch.knn.index.query.SegmentLevelQuantizationInfo.build(reader, fieldInfo, FIELD_NAME)
            ).thenReturn(quantizationInfo);
            quantizationUtilMockedStatic.when(
                () -> org.opensearch.knn.index.query.SegmentLevelQuantizationUtil.isAdcEnabled(quantizationInfo)
            ).thenReturn(true);
            quantizationUtilMockedStatic.when(
                () -> org.opensearch.knn.index.query.SegmentLevelQuantizationUtil.transformVectorWithADC(
                    queryVector,
                    quantizationInfo,
                    spaceType
                )
            ).thenAnswer(invocation -> null);

            final KNNVectorValues knnFloatVectorValues = TestVectorValues.createKNNFloatVectorValues(floatVectors);
            final KNNVectorValues knnByteVectorValues = TestVectorValues.createKNNBinaryVectorValues(quantizedDataVectors);
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader))
                .thenReturn(knnFloatVectorValues);
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader, true))
                .thenReturn(knnByteVectorValues);

            TopDocs docs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContext);
            assertEquals(quantizedDataVectors.size(), docs.scoreDocs.length);

            quantizationUtilMockedStatic.verify(
                () -> org.opensearch.knn.index.query.SegmentLevelQuantizationUtil.transformVectorWithADC(
                    queryVector,
                    quantizationInfo,
                    spaceType
                )
            );

            final List<Float> expectedScores = quantizedDataVectors.stream()
                .map(quantizedVector -> KNNScoringUtil.scoreWithADC(queryVector, quantizedVector, spaceType))
                .sorted(Comparator.reverseOrder())
                .toList();
            final List<Float> actualScores = Arrays.stream(docs.scoreDocs).map(scoreDoc -> scoreDoc.score).toList();
            assertEquals(expectedScores, actualScores);
        }
    }

    @SneakyThrows
    private void doTestRadialSearch_whenNoEngineFiles_thenSuccess(final boolean memoryOptimizedSearchEnabled) {
        try (MockedStatic<KNNVectorValuesFactory> valuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            // Prepare data
            final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
            final SpaceType spaceType = randomFrom(SpaceType.L2, SpaceType.INNER_PRODUCT);
            final List<float[]> dataVectors = Arrays.asList(
                new float[] { 11.0f, 12.0f, 13.0f },
                new float[] { 14.0f, 15.0f, 16.0f },
                new float[] { 17.0f, 18.0f, 19.0f }
            );
            final List<Float> expectedScores = dataVectors.stream()
                .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector))
                .sorted(Comparator.reverseOrder())
                .collect(Collectors.toList());
            final Float score = Collections.min(expectedScores);
            // Since memory optimized searching relies on Lucene's score framework, we can use minScore as a radius without having to
            // convert
            // it. We should not convert it as it treats minScore as a distance.
            final float radius = memoryOptimizedSearchEnabled ? score : KNNEngine.FAISS.scoreToRadialThreshold(score, spaceType);
            final int maxResults = 1000;
            final KNNQuery.Context context = mock(KNNQuery.Context.class);
            when(context.getMaxResultWindow()).thenReturn(maxResults);
            KNNWeight.initialize(null);

            // Create exact search context
            final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
                ExactSearcher.ExactSearcherContext.builder()
                    // setting to true, so that if quantization details are present we want to do search on the quantized
                    // vectors as this flow is used in first pass of search.
                    .useQuantizedVectorsForSearch(false)
                    .floatQueryVector(queryVector)
                    .radius(radius)
                    .isMemoryOptimizedSearchEnabled(memoryOptimizedSearchEnabled)
                    .maxResultWindow(maxResults)
                    .field(FIELD_NAME);

            // Create exact searcher
            ExactSearcher exactSearcher = new ExactSearcher(null);
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);

            // Set up segment + Lucene Directory
            final FSDirectory directory = mock(FSDirectory.class);
            when(reader.directory()).thenReturn(directory);
            final SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                Version.LATEST,
                Version.LATEST,
                SEGMENT_NAME,
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
            when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);

            // Mocking field infos
            final Path path = mock(Path.class);
            when(directory.getDirectory()).thenReturn(path);
            final FieldInfos fieldInfos = mock(FieldInfos.class);
            final FieldInfo fieldInfo = mock(FieldInfo.class);
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(any())).thenReturn(fieldInfo);
            when(fieldInfo.attributes()).thenReturn(
                Map.of(
                    SPACE_TYPE,
                    spaceType.getValue(),
                    KNN_ENGINE,
                    KNNEngine.FAISS.getName(),
                    PARAMETERS,
                    String.format(Locale.ROOT, "{\"%s\":\"%s\"}", INDEX_DESCRIPTION_PARAMETER, "HNSW32")
                )
            );
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(spaceType.getValue());

            // Mocking float vector values
            KNNFloatVectorValues floatVectorValues = mock(KNNFloatVectorValues.class);
            DocIdSetIterator docIdSetIterator = mock(DocIdSetIterator.class);
            KNNVectorValuesIterator knnVectorValuesIterator = mock(KNNVectorValuesIterator.class);
            when(knnVectorValuesIterator.getDocIdSetIterator()).thenReturn(docIdSetIterator);
            when(floatVectorValues.getVectorValuesIterator()).thenReturn(knnVectorValuesIterator);
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenReturn(floatVectorValues);
            when(floatVectorValues.nextDoc()).thenReturn(0, 1, 2, NO_MORE_DOCS);
            when(floatVectorValues.getVector()).thenReturn(dataVectors.get(0), dataVectors.get(1), dataVectors.get(2));

            // Now, perform exact search and do a validation
            final TopDocs topDocs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
            assertEquals(topDocs.scoreDocs.length, dataVectors.size());
            List<Float> actualScores = Arrays.stream(topDocs.scoreDocs).map(scoreDoc -> scoreDoc.score).toList();
            assertEquals(expectedScores, actualScores);
        }
    }
}
