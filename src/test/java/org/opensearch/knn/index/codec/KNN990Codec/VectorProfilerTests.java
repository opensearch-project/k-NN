/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.index.SegmentWriteState;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.test.OpenSearchTestCase;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.mockito.Mockito.*;

public class VectorProfilerTests extends OpenSearchTestCase {

    @Mock
    private SegmentWriteState segmentWriteState;


    private AutoCloseable mocks;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        mocks = MockitoAnnotations.openMocks(this);
        VectorProfiler.clearSegmentContexts();
    }

    @After
    public void tearDown() throws Exception {
        super.tearDown();
        mocks.close();
        VectorProfiler.clearSegmentContexts();
    }

    @Test
    public void testSingletonBehavior() {
        // Verify singleton instance handling
        VectorProfiler instance1 = VectorProfiler.getInstance();
        VectorProfiler instance2 = VectorProfiler.getInstance();
        assertSame(instance1, instance2);

        // Test instance override for testing
        VectorProfiler mockInstance = mock(VectorProfiler.class);
        VectorProfiler.setInstance(mockInstance);
        assertSame(mockInstance, VectorProfiler.getInstance());
    }

    @Test
    public void testCalculateVector_HappyPath() {
        // Test valid mean calculation
        List<float[]> vectors = Arrays.asList(
                new float[]{1.0f, 2.0f},
                new float[]{3.0f, 4.0f}
        );

        float[] result = VectorProfiler.calculateVector(vectors, StatisticalOperators.MEAN);
        assertArrayEquals(new float[]{2.0f, 3.0f}, result, 0.001f);
    }

    @Test
    public void testCalculateVector_EmptyVectors() {
        // Test empty input handling
        Exception ex = assertThrows(IllegalArgumentException.class,
                () -> VectorProfiler.calculateVector(Collections.emptyList(), StatisticalOperators.MEAN));
        assertEquals("Vectors collection cannot be null or empty", ex.getMessage());
    }

    @Test
    public void testCalculateVector_DimensionMismatch() {
        // Test invalid input dimensions
        List<float[]> vectors = Arrays.asList(
                new float[]{1.0f, 2.0f},
                new float[]{3.0f}
        );

        Exception ex = assertThrows(IllegalArgumentException.class,
                () -> VectorProfiler.calculateVector(vectors, StatisticalOperators.MEAN));
        assertEquals("All vectors must have same dimension", ex.getMessage());
    }

    @Test
    public void testRecordReadTimeVectors_NewSegment() {
        // Test initial recording for a segment
        String segName = "test_segment";
        Path dirPath = Paths.get("/test/path");
        List<float[]> vectors = Arrays.asList(new float[]{1.0f, 2.0f});

        VectorProfiler.recordReadTimeVectors(segName, "v1", dirPath, vectors, StatisticalOperators.MEAN);

        String contextKey = segName + "_v1";
        assertEquals(1L, VectorProfiler.getSegmentVectorCount(contextKey));
        assertEquals(1, VectorProfiler.getSampleVectorsForSegment(contextKey).size());
    }

    @Test
    public void testRecordReadTimeVectors_ExistingSegment() {
        // Test cumulative recording
        String segName = "test_segment";
        Path dirPath = Paths.get("/test/path");

        // First recording
        VectorProfiler.recordReadTimeVectors(segName, "v1", dirPath,
                Arrays.asList(new float[]{1.0f, 2.0f}), StatisticalOperators.MEAN);

        // Second recording
        VectorProfiler.recordReadTimeVectors(segName, "v1", dirPath,
                Arrays.asList(new float[]{3.0f, 4.0f}), StatisticalOperators.MEAN);

        String contextKey = segName + "_v1";
        assertEquals(2L, VectorProfiler.getSegmentVectorCount(contextKey));
        assertEquals(2, VectorProfiler.getSampleVectorsForSegment(contextKey).size());
    }

    @Test
    public void testRecordReadTimeVectors_EmptyVectors() {
        // Test null/empty input handling
        VectorProfiler.recordReadTimeVectors("empty_seg", "v1", Paths.get("/test"), null, StatisticalOperators.MEAN);
        VectorProfiler.recordReadTimeVectors("empty_seg", "v1", Paths.get("/test"), Collections.emptyList(), StatisticalOperators.MEAN);

        // Verify no data recorded
        String contextKey = "empty_seg_v1";
        assertEquals(0L, VectorProfiler.getSegmentVectorCount(contextKey));
    }

    @Test
    public void testAppendVector_ShortVector() {
        // Test full vector printing
        StringBuilder sb = new StringBuilder();
        float[] vector = {1.1f, 2.2f, 3.3f};
        VectorProfiler.appendVector(sb, vector);
        assertEquals("1.1, 2.2, 3.3", sb.toString());
    }

    @Test
    public void testAppendVector_LongVector() {
        // Test truncated vector printing
        StringBuilder sb = new StringBuilder();
        float[] vector = new float[25];
        Arrays.fill(vector, 1.0f);

        VectorProfiler.appendVector(sb, vector);
        assertTrue(sb.toString().contains("..."));
        assertTrue(sb.toString().length() < (25 * 6)); // 6 chars per element approx
    }

    @Test
    public void testClearSegmentContexts() {
        // Test context cleanup
        VectorProfiler.recordReadTimeVectors("seg1", "v1", Paths.get("/test"),
                Arrays.asList(new float[]{1.0f}), StatisticalOperators.MEAN);

        VectorProfiler.clearSegmentContexts();
        assertEquals(0, VectorProfiler.getSampleVectorsForSegment("seg1_v1").size());
    }


    @Test
    public void testGetSegmentMetadata() {
        // Test metadata accessors
        String segName = "meta_seg";
        Path dirPath = Paths.get("/meta/path");
        VectorProfiler.recordReadTimeVectors(segName, "v2", dirPath,
                Arrays.asList(new float[]{1.0f}), StatisticalOperators.MEAN);

        String contextKey = segName + "_v2";
        assertEquals(segName, VectorProfiler.getSegmentBaseName(contextKey));
        assertEquals("v2", VectorProfiler.getSegmentSuffix(contextKey));
        assertEquals(dirPath, VectorProfiler.getSegmentDirectoryPath(contextKey));
    }

    @Test
    public void testCalculateResultForSegment_NoVectors() {
        // Test empty segment handling
        String contextKey = "empty_seg_v1";
        assertNull(VectorProfiler.calculateResultForSegment(contextKey));
    }
}