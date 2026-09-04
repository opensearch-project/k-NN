/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.test.OpenSearchTestCase;

public class SvsLibraryTests extends OpenSearchTestCase {

    public void testCapabilityFlags() {
        assertTrue(SvsLibrary.INSTANCE.supportsIterativeBuild());
        assertTrue(SvsLibrary.INSTANCE.createsCustomSegmentFiles());
        assertTrue(SvsLibrary.INSTANCE.supportsFilters());
        assertTrue(SvsLibrary.INSTANCE.supportsRadialSearch());
        assertFalse(SvsLibrary.INSTANCE.supportsNestedFields());
    }

    public void testExtension() {
        assertEquals(".svs", SvsLibrary.INSTANCE.getExtension());
    }

    public void testScoreAndRadialThresholdsDelegateToFaiss() {
        for (SpaceType spaceType : new SpaceType[] { SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL }) {
            // inputs chosen so every space type converts to a positive radius
            float distance = spaceType == SpaceType.INNER_PRODUCT ? -0.5f : 0.5f;
            float score = spaceType == SpaceType.INNER_PRODUCT ? 1.5f : 0.75f;
            assertEquals(Faiss.INSTANCE.score(0.25f, spaceType), SvsLibrary.INSTANCE.score(0.25f, spaceType), 0.0f);
            assertEquals(
                Faiss.INSTANCE.distanceToRadialThreshold(distance, spaceType),
                SvsLibrary.INSTANCE.distanceToRadialThreshold(distance, spaceType)
            );
            assertEquals(
                Faiss.INSTANCE.scoreToRadialThreshold(score, spaceType),
                SvsLibrary.INSTANCE.scoreToRadialThreshold(score, spaceType)
            );
        }
    }

    public void testRadialThresholds_whenConvertedRadiusNonPositive_thenThrow() {
        // cosine min_score 0.3 converts to 2*0.3-1 = -0.4; inner-product max_distance 0 converts to 0.
        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> SvsLibrary.INSTANCE.scoreToRadialThreshold(0.3f, SpaceType.COSINESIMIL)
        );
        assertTrue(e.getMessage().contains("non-positive radius"));
        e = expectThrows(
            IllegalArgumentException.class,
            () -> SvsLibrary.INSTANCE.distanceToRadialThreshold(0.0f, SpaceType.INNER_PRODUCT)
        );
        assertTrue(e.getMessage().contains("non-positive radius"));
        assertEquals(
            Faiss.INSTANCE.scoreToRadialThreshold(0.9f, SpaceType.COSINESIMIL),
            SvsLibrary.INSTANCE.scoreToRadialThreshold(0.9f, SpaceType.COSINESIMIL)
        );
    }
}
