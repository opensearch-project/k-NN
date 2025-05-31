/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

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
            ExactSearcher.ExactSearcherContext.builder().knnQuery(query);

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
            ExactSearcher.ExactSearcherContext.builder().knnQuery(query);

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
        try (MockedStatic<KNNVectorValuesFactory> valuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
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
            final float radius = KNNEngine.FAISS.scoreToRadialThreshold(score, spaceType);
            final int maxResults = 1000;
            final KNNQuery.Context context = mock(KNNQuery.Context.class);
            when(context.getMaxResultWindow()).thenReturn(maxResults);
            KNNWeight.initialize(null);

            final KNNQuery query = KNNQuery.builder()
                .field(FIELD_NAME)
                .queryVector(queryVector)
                .radius(radius)
                .indexName(INDEX_NAME)
                .vectorDataType(VectorDataType.FLOAT)
                .context(context)
                .build();

            final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
                ExactSearcher.ExactSearcherContext.builder()
                    // setting to true, so that if quantization details are present we want to do search on the quantized
                    // vectors as this flow is used in first pass of search.
                    .useQuantizedVectorsForSearch(false)
                    .knnQuery(query);

            ExactSearcher exactSearcher = new ExactSearcher(null);
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);

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
            KNNFloatVectorValues floatVectorValues = mock(KNNFloatVectorValues.class);
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenReturn(floatVectorValues);
            when(floatVectorValues.nextDoc()).thenReturn(0, 1, 2, NO_MORE_DOCS);
            when(floatVectorValues.getVector()).thenReturn(dataVectors.get(0), dataVectors.get(1), dataVectors.get(2));
            final TopDocs topDocs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
            assertEquals(topDocs.scoreDocs.length, dataVectors.size());
            List<Float> actualScores = Arrays.stream(topDocs.scoreDocs).map(scoreDoc -> scoreDoc.score).toList();
            assertEquals(expectedScores, actualScores);
        }
    }
}
