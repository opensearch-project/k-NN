/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.Version;
import org.opensearch.common.collect.Tuple;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

public class KNN1040PerFieldKnnVectorsFormatTests extends KNNTestCase {

    public void testToTinySegmentsThreshold_whenNegativeOne_thenReturnsIntegerMaxValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(-1));
    }

    public void testToTinySegmentsThreshold_whenZero_thenReturnsZero() {
        // 0 → 0 → docCount < 0 is never true → always build the graph (matches Faiss semantics).
        assertEquals(0, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(0));
    }

    public void testToTinySegmentsThreshold_whenPositive_thenReturnsSameValue() {
        assertEquals(500, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(500));
        assertEquals(100, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(100));
        assertEquals(1, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(1));
    }

    public void testToTinySegmentsThreshold_whenLargeNegative_thenReturnsIntegerMaxValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(Integer.MIN_VALUE));
    }

    public void testToTinySegmentsThreshold_whenIntegerMaxValue_thenReturnsSameValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(Integer.MAX_VALUE));
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsOne_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(1);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsZero_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(0);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsNegative_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(-1);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyAboveOne_thenReturnsExecutorWithCorrectPoolSize() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        try {
            assertEquals(Integer.valueOf(4), result.v1());
            assertNotNull(result.v2());
            assertTrue(result.v2() instanceof ThreadPoolExecutor);
            ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
            assertEquals(4, executor.getCorePoolSize());
            assertEquals(4, executor.getMaximumPoolSize());
        } finally {
            result.v2().shutdownNow();
        }
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyAboveOne_thenExecutorAllowsCoreThreadTimeout() {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(2);
        try {
            ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
            assertTrue(executor.allowsCoreThreadTimeOut());
        } finally {
            result.v2().shutdownNow();
        }
    }

    public void testBuildMergeThreadCountAndExecutorService_whenCalledMultipleTimes_thenReturnsIndependentExecutors() {
        Tuple<Integer, ExecutorService> first = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        Tuple<Integer, ExecutorService> second = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        try {
            assertNotSame(first.v2(), second.v2());
        } finally {
            first.v2().shutdownNow();
            second.v2().shutdownNow();
        }
    }

    public void testThreadsCulledAfterTimeout() throws Exception {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(
            4,
            1L,
            TimeUnit.SECONDS
        );
        ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
        try {
            CountDownLatch tasksStarted = new CountDownLatch(4);
            CountDownLatch tasksCanFinish = new CountDownLatch(1);
            for (int i = 0; i < 4; i++) {
                executor.submit(() -> {
                    tasksStarted.countDown();
                    try {
                        tasksCanFinish.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            }
            assertTrue(tasksStarted.await(5, TimeUnit.SECONDS));
            assertEquals(4, executor.getActiveCount());

            tasksCanFinish.countDown();
            assertBusy(() -> assertEquals(0, executor.getPoolSize()), 10, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }
    }

    public void testThreadsSurviveDuringActiveMerge() throws Exception {
        Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(
            4,
            1L,
            TimeUnit.SECONDS
        );
        ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
        try {
            CountDownLatch tasksStarted = new CountDownLatch(4);
            CountDownLatch tasksCanFinish = new CountDownLatch(1);
            for (int i = 0; i < 4; i++) {
                executor.submit(() -> {
                    tasksStarted.countDown();
                    try {
                        tasksCanFinish.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            }
            assertTrue(tasksStarted.await(5, TimeUnit.SECONDS));

            Thread.sleep(2000);
            assertEquals(4, executor.getPoolSize());
            assertEquals(4, executor.getActiveCount());

            tasksCanFinish.countDown();
        } finally {
            executor.shutdownNow();
        }
    }

    public void testExecutorIsGarbageCollectedAfterUse() throws Exception {
        List<WeakReference<ExecutorService>> refs = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Tuple<Integer, ExecutorService> result = KNN1040PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(
                2,
                1L,
                TimeUnit.SECONDS
            );
            refs.add(new WeakReference<>(result.v2()));
        }

        Thread.sleep(3000);

        for (int i = 0; i < 10; i++) {
            System.gc();
            Thread.sleep(100);
        }

        long collected = refs.stream().filter(ref -> ref.get() == null).count();
        assertTrue("Expected at least one executor to be GC'd, but none were", collected > 0);
    }

    // --- FLAT format selection: compression level and data type pick the format and its encoding ---

    public void testGetKnnVectorsFormatForField_whenHalfFloatFlat_thenCompressionSelectsFormat() {
        assertEquals(KNN1040HalfFloatFlatVectorsFormat.class, resolveFlatFormat(VectorDataType.HALF_FLOAT, CompressionLevel.x1).getClass());
        assertFlatScalarQuantizedFormat(
            VectorDataType.HALF_FLOAT,
            CompressionLevel.x16,
            KNN1040HalfFloatScalarQuantizedVectorsFormat.class,
            ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE
        );
        assertFlatScalarQuantizedFormat(
            VectorDataType.HALF_FLOAT,
            CompressionLevel.x8,
            KNN1040HalfFloatScalarQuantizedVectorsFormat.class,
            ScalarEncoding.DIBIT_QUERY_NIBBLE
        );
        assertFlatScalarQuantizedFormat(
            VectorDataType.HALF_FLOAT,
            CompressionLevel.x4,
            KNN1040HalfFloatScalarQuantizedVectorsFormat.class,
            ScalarEncoding.PACKED_NIBBLE
        );
    }

    public void testGetKnnVectorsFormatForField_whenFloatFlat_thenCompressionSelectsEncoding() {
        assertFlatScalarQuantizedFormat(
            VectorDataType.FLOAT,
            CompressionLevel.x32,
            KNN1040ScalarQuantizedVectorsFormat.class,
            ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE
        );
        assertFlatScalarQuantizedFormat(
            VectorDataType.FLOAT,
            CompressionLevel.x16,
            KNN1040ScalarQuantizedVectorsFormat.class,
            ScalarEncoding.DIBIT_QUERY_NIBBLE
        );
        assertFlatScalarQuantizedFormat(
            VectorDataType.FLOAT,
            CompressionLevel.x8,
            KNN1040ScalarQuantizedVectorsFormat.class,
            ScalarEncoding.PACKED_NIBBLE
        );
    }

    // A level with no 1/2/4-bit width for the data type must fail loudly rather than silently
    // taking a default encoding.
    public void testGetKnnVectorsFormatForField_whenFloatFlatCompressionHasNoSQWidth_thenThrows() {
        // x4 leaves a float 8 bits: the Lucene 7-bit path, which the flat method does not offer.
        expectThrows(IllegalArgumentException.class, () -> resolveFlatFormat(VectorDataType.FLOAT, CompressionLevel.x4));
        // NOT_CONFIGURED is 32 bits; the flat resolver always resolves a level, so this never reaches the codec.
        expectThrows(IllegalArgumentException.class, () -> resolveFlatFormat(VectorDataType.FLOAT, CompressionLevel.NOT_CONFIGURED));
    }

    private void assertFlatScalarQuantizedFormat(
        VectorDataType vectorDataType,
        CompressionLevel compressionLevel,
        Class<? extends KnnVectorsFormat> expectedClass,
        ScalarEncoding expectedEncoding
    ) {
        KnnVectorsFormat format = resolveFlatFormat(vectorDataType, compressionLevel);
        assertEquals(vectorDataType + " at " + compressionLevel, expectedClass, format.getClass());
        assertTrue(format.toString(), format.toString().contains("encoding=" + expectedEncoding));
    }

    private KnnVectorsFormat resolveFlatFormat(VectorDataType vectorDataType, CompressionLevel compressionLevel) {
        KNNMethodContext flatContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Collections.emptyMap())
        );
        ResolvedIndexSpec resolvedSpec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.LUCENE)
            .methodName(METHOD_FLAT)
            .vectorDataType(vectorDataType)
            .dimension(3)
            .compressionLevel(compressionLevel)
            .indexVersionCreated(Version.CURRENT)
            .build();
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            "test_field",
            Collections.emptyMap(),
            vectorDataType,
            getMappingConfigForMethodMapping(flatContext, 3),
            Version.CURRENT,
            resolvedSpec
        );
        MapperService mapperService = mock(MapperService.class);
        when(mapperService.fieldType(eq("test_field"))).thenReturn(fieldType);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(null);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);
        return new KNN1040PerFieldKnnVectorsFormat(Optional.of(mapperService)).getKnnVectorsFormatForField("test_field");
    }
}
