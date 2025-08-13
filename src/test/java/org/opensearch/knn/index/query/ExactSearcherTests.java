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
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.iterators.BinaryVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.KNNIterator;
import org.opensearch.knn.index.query.iterators.NestedVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.VectorIdsKNNIterator;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
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
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

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

    @SneakyThrows
    public void testCreateIterator_whenNoVectorField_thenReturnsNull() {
        ExactSearcher exactSearcher = new ExactSearcher(null);
        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        SegmentReader reader = mock(SegmentReader.class);
        when(leafReaderContext.reader()).thenReturn(reader);

        FieldInfos fieldInfos = mock(FieldInfos.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(FIELD_NAME)).thenReturn(null);

        ExactSearcher.ExactSearcherContext context = ExactSearcher.ExactSearcherContext.builder()
            .field(FIELD_NAME)
            .floatQueryVector(new float[] { 1.0f, 0.0f })
            .build();

        KNNIterator iterator = exactSearcher.createIterator(leafReaderContext, context);
        assertNull(iterator);
    }

    @SneakyThrows
    public void testCreateIterator_withFloatVectors_thenReturnsVectorIterator() {
        try (MockedStatic<KNNVectorValuesFactory> factoryMock = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            ExactSearcher exactSearcher = new ExactSearcher(null);
            LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);
            SpaceType spaceType = SpaceType.L2;

            FieldInfos fieldInfos = mock(FieldInfos.class);
            FieldInfo fieldInfo = mock(FieldInfo.class);
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(FIELD_NAME)).thenReturn(fieldInfo);
            when(fieldInfo.attributes()).thenReturn(Map.of(SPACE_TYPE, spaceType.getValue(), KNN_ENGINE, KNNEngine.FAISS.getName()));
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(spaceType.getValue());

            KNNFloatVectorValues vectorValues = mock(KNNFloatVectorValues.class);
            factoryMock.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenReturn(vectorValues);

            ExactSearcher.ExactSearcherContext context = ExactSearcher.ExactSearcherContext.builder()
                .field(FIELD_NAME)
                .floatQueryVector(new float[] { 1.0f, 0.0f })
                .build();

            KNNIterator iterator = exactSearcher.createIterator(leafReaderContext, context);
            assertNotNull(iterator);
            assertTrue(iterator instanceof VectorIdsKNNIterator);
        }
    }

    @SneakyThrows
    public void testCreateIterator_withNestedQuery_thenReturnsNestedIterator() {
        try (MockedStatic<KNNVectorValuesFactory> factoryMock = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            ExactSearcher exactSearcher = new ExactSearcher(null);
            LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);
            SpaceType spaceType = SpaceType.L2;

            FieldInfos fieldInfos = mock(FieldInfos.class);
            FieldInfo fieldInfo = mock(FieldInfo.class);
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(FIELD_NAME)).thenReturn(fieldInfo);
            when(fieldInfo.attributes()).thenReturn(Map.of(SPACE_TYPE, spaceType.getValue(), KNN_ENGINE, KNNEngine.FAISS.getName()));
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(spaceType.getValue());

            KNNFloatVectorValues vectorValues = mock(KNNFloatVectorValues.class);
            factoryMock.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenReturn(vectorValues);

            BitSetProducer parentFilter = mock(BitSetProducer.class);
            BitSet parentBitSet = mock(BitSet.class);
            when(parentFilter.getBitSet(leafReaderContext)).thenReturn(parentBitSet);

            ExactSearcher.ExactSearcherContext context = ExactSearcher.ExactSearcherContext.builder()
                .field(FIELD_NAME)
                .floatQueryVector(new float[] { 1.0f, 0.0f })
                .parentsFilter(parentFilter)
                .build();

            KNNIterator iterator = exactSearcher.createIterator(leafReaderContext, context);

            assertNotNull(iterator);
            assertTrue(iterator instanceof NestedVectorIdsKNNIterator);
        }
    }

    @SneakyThrows
    public void testCreateIterator_withUserDefinedOrDefaultSpaceType() {
        try (MockedStatic<KNNVectorValuesFactory> factoryMock = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            ExactSearcher exactSearcher = new ExactSearcher(null);
            LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);

            // field info has l2 space type (default)
            FieldInfos fieldInfos = mock(FieldInfos.class);
            FieldInfo fieldInfo = mock(FieldInfo.class);
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(FIELD_NAME)).thenReturn(fieldInfo);
            when(fieldInfo.attributes()).thenReturn(Map.of(SPACE_TYPE, SpaceType.L2.getValue(), KNN_ENGINE, KNNEngine.FAISS.getName()));
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());

            float[] queryVector = { 1.0f, 0.0f };
            float[] docVector = { 0.0f, 1.0f };

            KNNFloatVectorValues vectorValues = mock(KNNFloatVectorValues.class);
            when(vectorValues.nextDoc()).thenReturn(0, NO_MORE_DOCS);
            when(vectorValues.getVector()).thenReturn(docVector);

            factoryMock.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenReturn(vectorValues);

            // use field's l2 space type
            ExactSearcher.ExactSearcherContext contextL2 = ExactSearcher.ExactSearcherContext.builder()
                .field(FIELD_NAME)
                .floatQueryVector(queryVector)
                .build();

            VectorIdsKNNIterator iteratorL2 = (VectorIdsKNNIterator) exactSearcher.createIterator(leafReaderContext, contextL2);
            iteratorL2.nextDoc();
            float scoreL2 = iteratorL2.score();

            // reset mocks
            when(vectorValues.nextDoc()).thenReturn(0, NO_MORE_DOCS);
            when(vectorValues.getVector()).thenReturn(docVector);

            // use user-defined inner product space type
            ExactSearcher.ExactSearcherContext contextIP = ExactSearcher.ExactSearcherContext.builder()
                .field(FIELD_NAME)
                .floatQueryVector(queryVector)
                .exactKNNSpaceType(SpaceType.INNER_PRODUCT.getValue())
                .build();

            VectorIdsKNNIterator iteratorIP = (VectorIdsKNNIterator) exactSearcher.createIterator(leafReaderContext, contextIP);
            iteratorIP.nextDoc();
            float scoreIP = iteratorIP.score();

            assertNotEquals(scoreL2, scoreIP, 0.001f);

            float expectedL2 = SpaceType.L2.getKnnVectorSimilarityFunction().compare(queryVector, docVector);
            float expectedIP = SpaceType.INNER_PRODUCT.getKnnVectorSimilarityFunction().compare(queryVector, docVector);

            assertEquals(expectedL2, scoreL2, 0.001f);
            assertEquals(expectedIP, scoreIP, 0.001f);
        }
    }

    @SneakyThrows
    public void testCreateIterator_withBinaryVectors_thenReturnsBinaryIterator() {
        try (MockedStatic<KNNVectorValuesFactory> factoryMock = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            ExactSearcher exactSearcher = new ExactSearcher(null);
            LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);

            FieldInfos fieldInfos = mock(FieldInfos.class);
            FieldInfo fieldInfo = mock(FieldInfo.class);
            when(reader.getFieldInfos()).thenReturn(fieldInfos);
            when(fieldInfos.fieldInfo(FIELD_NAME)).thenReturn(fieldInfo);
            when(fieldInfo.attributes()).thenReturn(
                Map.of(
                    SPACE_TYPE,
                    SpaceType.HAMMING.getValue(),
                    KNN_ENGINE,
                    KNNEngine.FAISS.getName(),
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.BINARY.getValue()
                )
            );
            when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.HAMMING.getValue());
            when(fieldInfo.getAttribute(VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BINARY.getValue());
            when(fieldInfo.getAttribute(KNNConstants.KNN_ENGINE)).thenReturn(KNNEngine.FAISS.getName());

            KNNBinaryVectorValues vectorValues = mock(KNNBinaryVectorValues.class);
            factoryMock.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenReturn(vectorValues);

            ExactSearcher.ExactSearcherContext context = ExactSearcher.ExactSearcherContext.builder()
                .field(FIELD_NAME)
                .byteQueryVector(new byte[] { 1, 0 })
                .build();

            KNNIterator iterator = exactSearcher.createIterator(leafReaderContext, context);

            assertNotNull(iterator);
            assertTrue(iterator instanceof BinaryVectorIdsKNNIterator);
        }
    }
}
