/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.mockito.MockedStatic;
import org.opensearch.common.io.PathUtils;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.jni.JNIService;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.KNNRestTestCase.INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.index.KNNSettings.QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES;
import static org.opensearch.knn.index.KNNSettings.QUANTIZATION_STATE_CACHE_SIZE_LIMIT;

public class KNNWeightTestCase extends KNNTestCase {

    protected static final String FIELD_NAME = "target_field";
    protected static final float[] QUERY_VECTOR = new float[] { 1.8f, 2.4f };
    protected static final byte[] BYTE_QUERY_VECTOR = new byte[] { 1, 2 };
    protected static final String SEGMENT_NAME = "0";
    protected static final int K = 5;
    protected static final Set<String> SEGMENT_FILES_NMSLIB = Set.of("_0.cfe", "_0_2011_target_field.hnswc");
    protected static final Set<String> SEGMENT_FILES_FAISS = Set.of("_0.cfe", "_0_2011_target_field.faissc");
    protected static final Set<String> SEGMENT_FILES_DEFAULT = SEGMENT_FILES_FAISS;
    protected static final Set<String> SEGMENT_MULTI_FIELD_FILES_FAISS = Set.of(
        "_0.cfe",
        "_0_2011_target_field.faissc",
        "_0_2011_long_target_field.faissc"
    );
    protected static final String CIRCUIT_BREAKER_LIMIT_100KB = "100Kb";
    protected static final Integer EF_SEARCH = 10;
    protected static final Map<String, ?> HNSW_METHOD_PARAMETERS = Map.of(METHOD_PARAMETER_EF_SEARCH, EF_SEARCH);
    protected static final Map<Integer, Float> DOC_ID_TO_SCORES = Map.of(10, 0.4f, 101, 0.05f, 100, 0.8f, 50, 0.52f);
    protected static final Map<Integer, Float> FILTERED_DOC_ID_TO_SCORES = Map.of(101, 0.05f, 100, 0.8f, 50, 0.52f);
    protected static final Map<Integer, Float> EXACT_SEARCH_DOC_ID_TO_SCORES = Map.of(0, 0.12048191f);
    protected static final Map<Integer, Float> BINARY_EXACT_SEARCH_DOC_ID_TO_SCORES = Map.of(0, 0.5f);
    protected static final Query FILTER_QUERY = new TermQuery(new Term("foo", "fooValue"));
    protected static MockedStatic<NativeMemoryCacheManager> nativeMemoryCacheManagerMockedStatic;
    protected static MockedStatic<JNIService> jniServiceMockedStatic;

    protected static MockedStatic<KNNSettings> knnSettingsMockedStatic;

    @BeforeClass
    public static void setUpClass() throws Exception {
        final KNNSettings knnSettings = mock(KNNSettings.class);
        knnSettingsMockedStatic = mockStatic(KNNSettings.class);
        when(knnSettings.getSettingValue(eq(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED))).thenReturn(true);
        when(knnSettings.getSettingValue(eq(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT))).thenReturn(CIRCUIT_BREAKER_LIMIT_100KB);
        when(knnSettings.getSettingValue(eq(KNNSettings.KNN_CACHE_ITEM_EXPIRY_ENABLED))).thenReturn(false);
        when(knnSettings.getSettingValue(eq(KNNSettings.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES))).thenReturn(TimeValue.timeValueMinutes(10));

        final ByteSizeValue v = ByteSizeValue.parseBytesSizeValue(
            CIRCUIT_BREAKER_LIMIT_100KB,
            KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT
        );
        knnSettingsMockedStatic.when(KNNSettings::getClusterCbLimit).thenReturn(v);
        knnSettingsMockedStatic.when(KNNSettings::state).thenReturn(knnSettings);
        ByteSizeValue cacheSize = ByteSizeValue.parseBytesSizeValue("1024kb", QUANTIZATION_STATE_CACHE_SIZE_LIMIT); // Setting 1MB as an
        // example
        when(knnSettings.getSettingValue(eq(QUANTIZATION_STATE_CACHE_SIZE_LIMIT))).thenReturn(cacheSize);
        // Mock QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES setting
        TimeValue mockTimeValue = TimeValue.timeValueMinutes(10);
        when(knnSettings.getSettingValue(eq(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES))).thenReturn(mockTimeValue);

        nativeMemoryCacheManagerMockedStatic = mockStatic(NativeMemoryCacheManager.class);

        final NativeMemoryCacheManager nativeMemoryCacheManager = mock(NativeMemoryCacheManager.class);
        final NativeMemoryAllocation nativeMemoryAllocation = mock(NativeMemoryAllocation.class);
        when(nativeMemoryCacheManager.get(any(), anyBoolean())).thenReturn(nativeMemoryAllocation);

        nativeMemoryCacheManagerMockedStatic.when(NativeMemoryCacheManager::getInstance).thenReturn(nativeMemoryCacheManager);

        final MockedStatic<PathUtils> pathUtilsMockedStatic = mockStatic(PathUtils.class);
        final Path indexPath = mock(Path.class);
        when(indexPath.toString()).thenReturn("/mydrive/myfolder");
        pathUtilsMockedStatic.when(() -> PathUtils.get(anyString(), anyString())).thenReturn(indexPath);
    }

    @Before
    public void setupBeforeTest() {
        knnSettingsMockedStatic.when(() -> KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME)).thenReturn(0);
        knnSettingsMockedStatic.when(() -> KNNSettings.isKnnIndexFaissEfficientFilterExactSearchDisabled(INDEX_NAME)).thenReturn(false);
        jniServiceMockedStatic = mockStatic(JNIService.class);
    }

    @After
    public void tearDownAfterTest() {
        jniServiceMockedStatic.close();
    }

    protected Map<Integer, Float> getTranslatedScores(Function<Float, Float> scoreTranslator) {
        return DOC_ID_TO_SCORES.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getKey, entry -> scoreTranslator.apply(entry.getValue())));
    }

    protected KNNQueryResult[] getKNNQueryResults() {
        return DOC_ID_TO_SCORES.entrySet()
            .stream()
            .map(entry -> new KNNQueryResult(entry.getKey(), entry.getValue()))
            .collect(Collectors.toList())
            .toArray(new KNNQueryResult[0]);
    }

    protected KNNQueryResult[] getFilteredKNNQueryResults() {
        return FILTERED_DOC_ID_TO_SCORES.entrySet()
            .stream()
            .map(entry -> new KNNQueryResult(entry.getKey(), entry.getValue()))
            .collect(Collectors.toList())
            .toArray(new KNNQueryResult[0]);
    }

    protected SegmentReader mockSegmentReader() {
        return mockSegmentReader(true);
    }

    protected SegmentReader mockSegmentReader(boolean isCompoundFile) {
        Path path = mock(Path.class);

        FSDirectory directory = mock(FSDirectory.class);
        when(directory.getDirectory()).thenReturn(path);

        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            100,
            isCompoundFile,
            false,
            KNNCodecVersion.CURRENT_DEFAULT,
            Map.of(),
            new byte[StringHelper.ID_LENGTH],
            Map.of(),
            Sort.RELEVANCE
        );
        segmentInfo.setFiles(SEGMENT_FILES_FAISS);
        SegmentCommitInfo segmentCommitInfo = new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[StringHelper.ID_LENGTH]);

        SegmentReader reader = mock(SegmentReader.class);
        when(reader.directory()).thenReturn(directory);
        when(reader.getSegmentInfo()).thenReturn(segmentCommitInfo);
        return reader;
    }
}
