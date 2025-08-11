/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.util.concurrent.MoreExecutors;
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
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.After;
import org.junit.Before;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.threadpool.ThreadPool;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.KNNRestTestCase.FIELD_NAME;
import static org.opensearch.knn.KNNRestTestCase.INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.EXACT_SEARCH_THREAD_POOL;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class ExactSearcherTests extends KNNTestCase {

    private static final String SEGMENT_NAME = "0";
    private ExecutorService executor;

    @Before
    public void setExactSearchThreadPool() {
        ThreadPool threadPool = mock(ThreadPool.class);
        executor = MoreExecutors.newDirectExecutorService();
        when(threadPool.executor(EXACT_SEARCH_THREAD_POOL)).thenReturn(executor);
        ExactSearcher.initialize(threadPool);
    }

    @After
    public void shutdownExecutor() {
        executor.shutdown();
    }

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
        when(reader.maxDoc()).thenReturn(1);

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(query.getField())).thenReturn(null);
        TopDocs docs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
        Mockito.verify(fieldInfos).fieldInfo(query.getField());
        Mockito.verify(reader).getFieldInfos();
        Mockito.verify(leafReaderContext, times(2)).reader();
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
        when(reader.maxDoc()).thenReturn(1);

        final FieldInfos fieldInfos = mock(FieldInfos.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.fieldInfo(query.getField())).thenReturn(null);
        TopDocs docs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
        Mockito.verify(fieldInfos).fieldInfo(query.getField());
        Mockito.verify(reader).getFieldInfos();
        Mockito.verify(leafReaderContext, times(2)).reader();
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
                new float[] { 17.0f, 18.0f, 19.0f },
                new float[] { 20.0f, 21.0f, 22.0f },
                new float[] { 23.0f, 24.0f, 25.0f },
                new float[] { 26.0f, 27.0f, 28.0f }
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

            // Create exact searcher
            ExactSearcher exactSearcher = new ExactSearcher(null);
            ExactSearcher.setConcurrentExactSearchEnabled(true);
            ExactSearcher.setConcurrentExactSearchMinDocumentCount(2);
            ExactSearcher.setConcurrentExactSearchMaxPartitionCount(0);
            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);
            when(reader.maxDoc()).thenReturn(6);

            // Set up segment + Lucene Directory
            final FSDirectory directory = mock(FSDirectory.class);
            when(reader.directory()).thenReturn(directory);
            final SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                Version.LATEST,
                Version.LATEST,
                SEGMENT_NAME,
                6,
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
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenAnswer(invocation -> {
                KNNFloatVectorValues floatVectorValues = mock(KNNFloatVectorValues.class);
                AtomicInteger lastReturned = new AtomicInteger(-1);
                when(floatVectorValues.advance(anyInt())).thenAnswer(advInvocation -> {
                    int target = advInvocation.getArgument(0);
                    int prev = lastReturned.get();
                    assertTrue(prev < target);
                    int[] docs = { 0, 1, 2, 3, 4, 5, Integer.MAX_VALUE };
                    for (int doc : docs) {
                        if (doc >= target) {
                            lastReturned.set(doc);
                            return doc;
                        }
                    }
                    lastReturned.set(Integer.MAX_VALUE);
                    return Integer.MAX_VALUE;
                });
                when(floatVectorValues.getVector()).thenAnswer(vecInvocation -> dataVectors.get(lastReturned.get()));
                return floatVectorValues;
            });

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

            // Now, perform exact search and do a validation
            final TopDocs topDocs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
            assertEquals(topDocs.scoreDocs.length, dataVectors.size());
            List<Float> actualScores = Arrays.stream(topDocs.scoreDocs).map(scoreDoc -> scoreDoc.score).toList();
            assertEquals(expectedScores, actualScores);
        }
    }

    @SneakyThrows
    public void testExactSearch_withNestedField() {
        try (MockedStatic<KNNVectorValuesFactory> valuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            // Prepare data
            final SpaceType spaceType = randomFrom(SpaceType.L2, SpaceType.INNER_PRODUCT);
            final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
            int[] docs = { 0, 2, 3, 4, 6, 7, Integer.MAX_VALUE };
            final Map<Integer, float[]> dataVectors = Map.of(
                0,
                new float[] { 11.0f, 12.0f, 13.0f },
                2,
                new float[] { 14.0f, 15.0f, 16.0f },
                3,
                new float[] { 17.0f, 18.0f, 19.0f },
                4,
                new float[] { 20.0f, 21.0f, 22.0f },
                6,
                new float[] { 23.0f, 24.0f, 25.0f },
                7,
                new float[] { 26.0f, 27.0f, 28.0f }
            );

            final BitSet parentBitSet = new FixedBitSet(new long[] { 290 }, 9);
            final Map<Integer, Float> expectedScores = dataVectors.entrySet()
                .stream()
                .collect(
                    Collectors.toMap(
                        e -> parentBitSet.nextSetBit(e.getKey() + 1),
                        e -> Map.entry(e.getKey(), spaceType.getKnnVectorSimilarityFunction().compare(queryVector, e.getValue())),
                        (a, b) -> a.getValue() >= b.getValue() ? a : b
                    )
                )
                .values()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

            // Create exact searcher
            ExactSearcher exactSearcher = new ExactSearcher(null);
            ExactSearcher.setConcurrentExactSearchEnabled(true);
            ExactSearcher.setConcurrentExactSearchMinDocumentCount(2);
            ExactSearcher.setConcurrentExactSearchMaxPartitionCount(0);

            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);
            when(reader.maxDoc()).thenReturn(9);

            // Set up segment + Lucene Directory
            final FSDirectory directory = mock(FSDirectory.class);
            when(reader.directory()).thenReturn(directory);
            final SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                Version.LATEST,
                Version.LATEST,
                SEGMENT_NAME,
                9,
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

            // Mocking parent filter
            BitSetProducer parentsFilter = mock(BitSetProducer.class);
            when(parentsFilter.getBitSet(leafReaderContext)).thenReturn(parentBitSet);

            // Mocking float vector values
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenAnswer(invocation -> {
                KNNFloatVectorValues floatVectorValues = mock(KNNFloatVectorValues.class);
                AtomicInteger lastReturned = new AtomicInteger(-1);
                when(floatVectorValues.advance(anyInt())).thenAnswer(advInvocation -> {
                    int target = advInvocation.getArgument(0);
                    int prev = lastReturned.get();
                    assertTrue(prev < target);
                    for (int doc : docs) {
                        if (doc >= target) {
                            lastReturned.set(doc);
                            return doc;
                        }
                    }
                    lastReturned.set(Integer.MAX_VALUE);
                    return Integer.MAX_VALUE;
                });
                when(floatVectorValues.getVector()).thenAnswer(vecInvocation -> dataVectors.get(lastReturned.get()));
                return floatVectorValues;
            });

            // Create exact search context
            final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
                ExactSearcher.ExactSearcherContext.builder()
                    .parentsFilter(parentsFilter)
                    .useQuantizedVectorsForSearch(false)
                    .floatQueryVector(queryVector)
                    .isMemoryOptimizedSearchEnabled(false)
                    .k(1000)
                    .field(FIELD_NAME);

            // Now, perform exact search and do a validation
            final TopDocs topDocs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
            assertEquals(topDocs.scoreDocs.length, parentBitSet.cardinality());
            Map<Integer, Float> actualScores = Arrays.stream(topDocs.scoreDocs)
                .collect(Collectors.toMap(scoreDoc -> scoreDoc.doc, scoreDoc -> scoreDoc.score));
            assertEquals(expectedScores, actualScores);
        }
    }

    @SneakyThrows
    public void testExactSearch_withFilter() {
        try (MockedStatic<KNNVectorValuesFactory> valuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            // Prepare data
            final SpaceType spaceType = randomFrom(SpaceType.L2, SpaceType.INNER_PRODUCT);
            final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
            int[] docs = { 0, 1, 2, 3, 4, 5, Integer.MAX_VALUE };
            final List<float[]> dataVectors = Arrays.asList(
                new float[] { 11.0f, 12.0f, 13.0f },
                new float[] { 14.0f, 15.0f, 16.0f },
                new float[] { 17.0f, 18.0f, 19.0f },
                new float[] { 20.0f, 21.0f, 22.0f },
                new float[] { 23.0f, 24.0f, 25.0f },
                new float[] { 26.0f, 27.0f, 28.0f }
            );
            final int[] filterIds = { 0, 2, 3, 5 };
            final Set<Integer> filterSet = Arrays.stream(filterIds).boxed().collect(Collectors.toSet());
            final List<Float> expectedScores = IntStream.range(0, dataVectors.size())
                .filter(filterSet::contains)
                .mapToObj(i -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, dataVectors.get(i)))
                .sorted(Comparator.reverseOrder())
                .collect(Collectors.toList());
            FixedBitSet filterBitSet = new FixedBitSet(6);
            for (int id : filterIds) {
                filterBitSet.set(id);
            }

            // Create exact searcher
            ExactSearcher exactSearcher = new ExactSearcher(null);
            ExactSearcher.setConcurrentExactSearchEnabled(true);
            ExactSearcher.setConcurrentExactSearchMinDocumentCount(2);
            ExactSearcher.setConcurrentExactSearchMaxPartitionCount(0);

            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);
            when(reader.maxDoc()).thenReturn(9);

            // Set up segment + Lucene Directory
            final FSDirectory directory = mock(FSDirectory.class);
            when(reader.directory()).thenReturn(directory);
            final SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                Version.LATEST,
                Version.LATEST,
                SEGMENT_NAME,
                9,
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
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenAnswer(invocation -> {
                KNNFloatVectorValues floatVectorValues = mock(KNNFloatVectorValues.class);

                AtomicInteger lastReturned = new AtomicInteger(-1);
                when(floatVectorValues.advance(anyInt())).thenAnswer(advInvocation -> {
                    int target = advInvocation.getArgument(0);
                    int prev = lastReturned.get();
                    assertTrue(prev < target);
                    for (int doc : docs) {
                        if (doc >= target) {
                            lastReturned.set(doc);
                            return doc;
                        }
                    }
                    lastReturned.set(Integer.MAX_VALUE);
                    return Integer.MAX_VALUE;
                });
                when(floatVectorValues.getVector()).thenAnswer(inv -> dataVectors.get(lastReturned.get()));
                return floatVectorValues;
            });

            // Create exact search context
            final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
                ExactSearcher.ExactSearcherContext.builder()
                    .matchedDocs(filterBitSet)
                    .useQuantizedVectorsForSearch(false)
                    .floatQueryVector(queryVector)
                    .isMemoryOptimizedSearchEnabled(false)
                    .k(1000)
                    .field(FIELD_NAME);

            // Now, perform exact search and do a validation
            final TopDocs topDocs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
            assertEquals(topDocs.scoreDocs.length, filterIds.length);
            List<Float> actualScores = Arrays.stream(topDocs.scoreDocs).map(scoreDoc -> scoreDoc.score).toList();
            assertEquals(expectedScores, actualScores);
        }
    }

    @SneakyThrows
    public void testExactSearch_withAllDocs() {
        try (MockedStatic<KNNVectorValuesFactory> valuesFactoryMockedStatic = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            // Prepare data
            final SpaceType spaceType = randomFrom(SpaceType.L2, SpaceType.INNER_PRODUCT);
            final float[] queryVector = new float[] { 0.1f, 2.0f, 3.0f };
            int[] docs = { 0, 1, 2, 3, 4, 5, Integer.MAX_VALUE };
            final List<float[]> dataVectors = Arrays.asList(
                new float[] { 11.0f, 12.0f, 13.0f },
                new float[] { 14.0f, 15.0f, 16.0f },
                new float[] { 17.0f, 18.0f, 19.0f },
                new float[] { 20.0f, 21.0f, 22.0f },
                new float[] { 23.0f, 24.0f, 25.0f },
                new float[] { 26.0f, 27.0f, 28.0f }
            );
            final List<Float> expectedScores = dataVectors.stream()
                .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector))
                .sorted(Comparator.reverseOrder())
                .collect(Collectors.toList());

            // Create exact searcher
            ExactSearcher exactSearcher = new ExactSearcher(null);
            ExactSearcher.setConcurrentExactSearchEnabled(true);
            ExactSearcher.setConcurrentExactSearchMinDocumentCount(2);
            ExactSearcher.setConcurrentExactSearchMaxPartitionCount(0);

            final LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
            final SegmentReader reader = mock(SegmentReader.class);
            when(leafReaderContext.reader()).thenReturn(reader);
            when(reader.maxDoc()).thenReturn(9);

            // Set up segment + Lucene Directory
            final FSDirectory directory = mock(FSDirectory.class);
            when(reader.directory()).thenReturn(directory);
            final SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                Version.LATEST,
                Version.LATEST,
                SEGMENT_NAME,
                9,
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
            valuesFactoryMockedStatic.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader)).thenAnswer(invocation -> {
                KNNFloatVectorValues floatVectorValues = mock(KNNFloatVectorValues.class);

                AtomicInteger lastReturned = new AtomicInteger(-1);
                when(floatVectorValues.advance(anyInt())).thenAnswer(advInvocation -> {
                    int target = advInvocation.getArgument(0);
                    int prev = lastReturned.get();
                    assertTrue(prev < target);
                    for (int doc : docs) {
                        if (doc >= target) {
                            lastReturned.set(doc);
                            return doc;
                        }
                    }
                    lastReturned.set(Integer.MAX_VALUE);
                    return Integer.MAX_VALUE;
                });
                when(floatVectorValues.getVector()).thenAnswer(inv -> dataVectors.get(lastReturned.get()));
                return floatVectorValues;
            });

            // Create exact search context
            final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
                ExactSearcher.ExactSearcherContext.builder()
                    .useQuantizedVectorsForSearch(false)
                    .floatQueryVector(queryVector)
                    .isMemoryOptimizedSearchEnabled(false)
                    .k(1000)
                    .field(FIELD_NAME);

            // Now, perform exact search and do a validation
            final TopDocs topDocs = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContextBuilder.build());
            assertEquals(topDocs.scoreDocs.length, dataVectors.size());
            List<Float> actualScores = Arrays.stream(topDocs.scoreDocs).map(scoreDoc -> scoreDoc.score).toList();
            assertEquals(expectedScores, actualScores);
        }
    }
}
