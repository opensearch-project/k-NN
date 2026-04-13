/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.junit.Test;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.MemoryOptimizedSearchScoreConverter;

import static org.junit.Assert.assertEquals;

public class MemoryOptimizedSearchScoreConverterTests {
    @Test
    public void testConvertingFaissScoreToLuceneScore() {
        // For L2, Lucene uses the score itself during search, so it must be as it is.
        final float faissL2Score = 0.84F;
        assertEquals(MemoryOptimizedSearchScoreConverter.scoreToRadialThreshold(faissL2Score, SpaceType.L2), faissL2Score, 1e-6);

        // For inner product, likewise L2, Lucene uses score for internal scoring. so it must return what it was given.
        final float faissIpScore1 = 0.88F;
        assertEquals(
            MemoryOptimizedSearchScoreConverter.scoreToRadialThreshold(faissIpScore1, SpaceType.INNER_PRODUCT),
            faissIpScore1,
            1e-6
        );

        final float faissIpScore2 = 5.84F;
        assertEquals(
            MemoryOptimizedSearchScoreConverter.scoreToRadialThreshold(faissIpScore2, SpaceType.INNER_PRODUCT),
            faissIpScore2,
            1e-6
        );

        // For cosine, Lucene now uses COSINE scorer directly, so score passes through as-is.
        final float faissCosine1 = 0.77F;
        assertEquals(MemoryOptimizedSearchScoreConverter.scoreToRadialThreshold(faissCosine1, SpaceType.COSINESIMIL), faissCosine1, 1e-6);

        final float faissCosine2 = 0.5F;
        assertEquals(MemoryOptimizedSearchScoreConverter.scoreToRadialThreshold(faissCosine2, SpaceType.COSINESIMIL), faissCosine2, 1e-6);
    }

    @Test
    public void testConvertingFaissDistanceToLuceneScore() {
        // For L2, Lucene is using the formula, score = 1 / (1 + d)
        final float faissL2Distance = 35.123F;
        final float luceneL2Radius = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(faissL2Distance, SpaceType.L2);
        assertEquals(1 / (1 + faissL2Distance), luceneL2Radius, 1e-6);

        // For IP, the input distance is actually `inner product value`
        // We're expecting converted maximum inner product.
        final float faissIpDistance1 = -0.5F;
        final float luceneIpRadius1 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissIpDistance1,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(1 / (1 - faissIpDistance1), luceneIpRadius1, 1e-6);

        final float faissIpDistance2 = 0;
        final float luceneIpRadius2 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissIpDistance2,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(1, luceneIpRadius2, 1e-6);

        final float faissIpDistance3 = 5.5F;
        final float luceneIpRadius3 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissIpDistance3,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(1 + faissIpDistance3, luceneIpRadius3, 1e-6);

        // For cosine, distance = 1 - cosine(a,b). Lucene COSINE scorer uses (2 - distance) / 2.
        final float faissCosineDistance1 = 0;
        final float luceneCosineRadius1 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissCosineDistance1,
            SpaceType.COSINESIMIL
        );
        assertEquals((2 - 0) / 2F, luceneCosineRadius1, 1e-6);

        final float faissCosineDistance2 = 2;
        final float luceneCosineRadius2 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissCosineDistance2,
            SpaceType.COSINESIMIL
        );
        assertEquals((2 - 2) / 2F, luceneCosineRadius2, 1e-6);

        final float faissCosineDistance3 = 1.44F;
        final float luceneCosineRadius3 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissCosineDistance3,
            SpaceType.COSINESIMIL
        );
        assertEquals((2 - 1.44F) / 2F, luceneCosineRadius3, 1e-6);

        final float faissCosineDistance4 = 0.77F;
        final float luceneCosineRadius4 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissCosineDistance4,
            SpaceType.COSINESIMIL
        );
        assertEquals((2 - 0.77F) / 2F, luceneCosineRadius4, 1e-6);
    }
}
