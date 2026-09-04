/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.engine.EngineParameters;
import org.opensearch.knn.index.engine.NativeSearchParams;
import org.opensearch.test.OpenSearchTestCase;

/**
 * Pins the query-time rejection paths of {@link SvsNativeEngineService}. All of these throw before any
 * native call, so this suite runs without the SVS native library.
 */
public class SvsNativeEngineServiceTests extends OpenSearchTestCase {

    private final SvsNativeEngineService service = new SvsNativeEngineService();

    public void testQueryIndex_whenParentIds_thenNestedRejected() {
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> service.queryIndex(
                1L,
                NativeSearchParams.forTopK(new float[] { 1f }, 10, EngineParameters.EMPTY, null, 0, new int[] { 3, 7 })
            )
        );
        assertTrue(e.getMessage(), e.getMessage().contains("Nested fields are not supported"));
    }

    public void testRadiusQueryIndex_whenParentIds_thenNestedRejected() {
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> service.radiusQueryIndex(
                1L,
                NativeSearchParams.forRadial(new float[] { 1f }, 1.0f, 10000, EngineParameters.EMPTY, null, 0, new int[] { 3, 7 })
            )
        );
        assertTrue(e.getMessage(), e.getMessage().contains("Nested fields are not supported"));
    }

    /**
     * The SVS index only accepts a strictly positive faiss-domain radius; the subset of inner-product and
     * cosine thresholds that convert to radius &lt;= 0 must be rejected with a descriptive message rather
     * than an opaque native error.
     */
    public void testRadiusQueryIndex_whenNonPositiveRadius_thenRejected() {
        for (float radius : new float[] { 0.0f, -0.4f }) {
            UnsupportedOperationException e = expectThrows(
                UnsupportedOperationException.class,
                () -> service.radiusQueryIndex(
                    1L,
                    NativeSearchParams.forRadial(new float[] { 1f }, radius, 10000, EngineParameters.EMPTY, null, 0, null)
                )
            );
            assertTrue(e.getMessage(), e.getMessage().contains("non-positive radius"));
        }
    }
}
