/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.search.ScoreDoc;
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

        // For cosine though, score function between Faiss and Lucene is different
        // Faiss's score = (1 + inner_product_value) / 2
        // Since MAXIMUM_INNER_PRODUCT is being used in Lucene, we should get max_inner_product value which is 1.54.
        // inner_product_value = 2 * score - 1
        // max_inner_product_value = inner_product_value + 1 if inner_product_value >= 0 else 1 / -inner_product_value
        final float faissCosine1 = 0.77F;
        final float luceneRadius1 = MemoryOptimizedSearchScoreConverter.scoreToRadialThreshold(faissCosine1, SpaceType.COSINESIMIL);
        assertEquals(luceneRadius1, 1.54F, 1e-6);

        // When two vectors are perpendicular to each other.
        final float faissCosine2 = 0.5F;
        final float luceneRadius2 = MemoryOptimizedSearchScoreConverter.scoreToRadialThreshold(faissCosine2, SpaceType.COSINESIMIL);
        assertEquals(luceneRadius2, 1F, 1e-6);
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

        // For cosine, the input distance is actually `1 - inner product value (whose range is in [-1, 1])`
        final float faissCosineDistance1 = 0;
        final float luceneCosineRadius1 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissCosineDistance1,
            SpaceType.COSINESIMIL
        );
        // inner product value is 1 from distance `0`, hence it should be 1 + the value.
        assertEquals(1 + 1, luceneCosineRadius1, 1e-6);

        final float faissCosineDistance2 = 2;
        final float luceneCosineRadius2 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissCosineDistance2,
            SpaceType.COSINESIMIL
        );
        // inner product value is -1, hence max inner product value should be 1 / (1 + 1)
        assertEquals(1F / (1 + 1), luceneCosineRadius2, 1e-6);

        final float faissCosineDistance3 = 1.44F;
        final float luceneCosineRadius3 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissCosineDistance3,
            SpaceType.COSINESIMIL
        );
        // inner product value is -0.44, hence max inner product value should be 1 / (1 + 0.44)
        assertEquals(1 / (1 + 0.44F), luceneCosineRadius3, 1e-6);

        final float faissCosineDistance4 = 0.77F;
        final float luceneCosineRadius4 = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(
            faissCosineDistance4,
            SpaceType.COSINESIMIL
        );
        // inner product value is 0.23, hence max inner product value should be 1 / (1 + 0.44)
        assertEquals(1.23F, luceneCosineRadius4, 1e-6);
    }

    @Test
    public void testConvertingLuceneCosineScoreToFaissScore() {
        // Actually, this is max inner product score value. For cosine, it is being used
        final float luceneCosineScore1 = 0.55F;
        final ScoreDoc[] scoreDoc1 = new ScoreDoc[] { new ScoreDoc(0, luceneCosineScore1) };
        MemoryOptimizedSearchScoreConverter.convertToCosineScore(scoreDoc1);
        // inner product value = 1 - 1 / 0.55, cosine score in Faiss = (2 - (1 - ip)) / 2 = (1 + ip) / 2
        assertEquals((1 + (1 - (1 / 0.55F))) / 2, scoreDoc1[0].score, 1e-6);

        final float luceneCosineScore2 = 1.55F;
        final ScoreDoc[] scoreDoc2 = new ScoreDoc[] { new ScoreDoc(0, luceneCosineScore2) };
        MemoryOptimizedSearchScoreConverter.convertToCosineScore(scoreDoc2);
        // inner product value = 1.55 - 1, cosine score in Faiss = (2 - (1 - ip)) / 2 = (1 + ip) / 2
        assertEquals((1 + (1.55 - 1)) / 2, scoreDoc2[0].score, 1e-6);
    }
}
