/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.rescore;

import org.opensearch.knn.KNNTestCase;

import static org.opensearch.knn.index.query.rescore.RescoreContext.MAX_FIRST_PASS_RESULTS;
import static org.opensearch.knn.index.query.rescore.RescoreContext.MIN_FIRST_PASS_RESULTS;

public class RescoreContextTests extends KNNTestCase {

    public void testGetFirstPassK_userProvidedOversample_notOverriddenByDimension() {
        float oversample = 2.6f;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(oversample).userProvided(true).build();
        int finalK = 100;

        assertEquals(260, rescoreContext.getFirstPassK(finalK, 500));
        assertEquals(260, rescoreContext.getFirstPassK(finalK, 1200));
    }

    public void testGetFirstPassK_boundsCheck_minAndMax() {
        float oversample = 2.6f;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(oversample).userProvided(true).build();

        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(1, 500));
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(0, 500));
        assertEquals(MAX_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(MAX_FIRST_PASS_RESULTS, 500));
    }

    public void testGetFirstPassK_dimensionAbove1000_oversampleIs1x() {
        int finalK = 100;
        int dimension = 1024;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(3.0f).userProvided(false).build();

        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK, dimension));
    }

    public void testGetFirstPassK_dimension768to999_oversampleIs2x() {
        int finalK = 100;
        int dimension = 800;
        // Regardless of initial oversampleFactor, dimension-based logic sets it to 2.0
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(3.0f).userProvided(false).build();

        assertEquals(200, rescoreContext.getFirstPassK(finalK, dimension));
    }

    public void testGetFirstPassK_dimensionBelow768_oversampleIs3x() {
        int finalK = 100;
        int dimension = 500;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(1.0f).userProvided(false).build();

        assertEquals(300, rescoreContext.getFirstPassK(finalK, dimension));
    }

    public void testGetFirstPassK_dimensionExactly768_oversampleIs2x() {
        int finalK = 100;
        int dimension = 768;
        RescoreContext rescoreContext = RescoreContext.builder().userProvided(false).build();

        assertEquals(200, rescoreContext.getFirstPassK(finalK, dimension));
    }

    public void testGetFirstPassK_dimensionExactly1000_oversampleIs1x() {
        int finalK = 100;
        int dimension = 1000;
        RescoreContext rescoreContext = RescoreContext.builder().userProvided(false).build();

        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK, dimension));
    }

    public void testGetFirstPassK_userProvided_neverOverriddenByDimension() {
        int finalK = 100;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(5.0f).userProvided(true).build();

        assertEquals(500, rescoreContext.getFirstPassK(finalK, 500));
        assertEquals(500, rescoreContext.getFirstPassK(finalK, 1200));
    }

    public void testGetFirstPassK_whenAllowOverrideIsFalse_thenDimensionBasedOverrideIsSkipped() {
        float oversample = RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR;
        int finalK = 100;

        RescoreContext rescoreContext = RescoreContext.builder()
            .oversampleFactor(oversample)
            .userProvided(false)
            .allowOverrideOversampleFactor(false)
            .build();

        // dimension < 768 would normally trigger 3x, but allowOverrideOversampleFactor=false keeps it at 2x
        assertEquals(200, rescoreContext.getFirstPassK(finalK, 500));

        rescoreContext = RescoreContext.builder()
            .oversampleFactor(oversample)
            .userProvided(false)
            .allowOverrideOversampleFactor(false)
            .build();
        // dimension >= 1000 would normally trigger 1x, but stays at 2x
        assertEquals(200, rescoreContext.getFirstPassK(finalK, 1200));
    }

    public void testGetFirstPassK_smallFinalK_clampedToMin() {
        int finalK = 10;
        int dimension = 700;
        RescoreContext rescoreContext = RescoreContext.builder().userProvided(false).build();

        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK, dimension));
    }

    @SuppressWarnings("deprecation")
    public void testGetFirstPassK_deprecatedMethodDelegatesToNewMethod() {
        int finalK = 100;
        int dimension = 500;
        RescoreContext rescoreContext = RescoreContext.builder().userProvided(false).build();

        // Deprecated 3-arg method should produce same result regardless of boolean value
        assertEquals(rescoreContext.getFirstPassK(finalK, dimension), rescoreContext.getFirstPassK(finalK, false, dimension));

        rescoreContext = RescoreContext.builder().userProvided(false).build();
        assertEquals(rescoreContext.getFirstPassK(finalK, dimension), rescoreContext.getFirstPassK(finalK, true, dimension));
    }
}
