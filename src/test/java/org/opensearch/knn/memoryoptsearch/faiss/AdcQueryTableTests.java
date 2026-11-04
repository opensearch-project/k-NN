/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class AdcQueryTableTests extends KNNTestCase {

    private static final SpaceType[] SUPPORTED = { SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL };

    /** Values every partial sum represents exactly in any order, so parity can be asserted exactly. */
    private static final float[] DYADIC = { -1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f, 1.0f };

    /**
     * One through forty covers every remainder the grouping can leave and dimensions below one group.
     * The rest are real index shapes, including odd group counts where the last nibble is read alone.
     */
    private static List<Integer> dimensions() {
        final List<Integer> dimensions = new ArrayList<>();
        for (int dimension = 1; dimension <= 40; dimension++) {
            dimensions.add(dimension);
        }
        for (int dimension : new int[] { 63, 64, 65, 100, 127, 128, 129, 256, 768, 772, 1023, 1024, 1536, 1540, 16000 }) {
            dimensions.add(dimension);
        }
        return dimensions;
    }

    /**
     * The property the class exists to hold. Exact, so a dropped or double counted dimension cannot
     * hide inside a tolerance.
     */
    @Test
    public void testExactlyMatchesScalarForRepresentableQueries() {
        for (int dimension : dimensions()) {
            for (SpaceType spaceType : SUPPORTED) {
                assertParity(dimension, spaceType, dyadicQuery(seedFor(dimension, spaceType), dimension), 128, 0.0f);
            }
        }
    }

    /**
     * A constant query makes every coefficient the same sign, so partial sums grow instead of
     * cancelling. Dyadic values stay exact; 0.1f does not, so it is the one that exercises growth.
     */
    @Test
    public void testMatchesScalarForConstantQuery() {
        for (int dimension : new int[] { 8, 13, 768, 772, 1536 }) {
            for (SpaceType spaceType : SUPPORTED) {
                for (float value : new float[] { 1.0f, -1.0f, 0.25f }) {
                    final float[] query = new float[dimension];
                    Arrays.fill(query, value);
                    assertParity(dimension, spaceType, query, 128, 0.0f);
                }
                final float[] growing = new float[dimension];
                Arrays.fill(growing, 0.1f);
                assertParity(dimension, spaceType, growing, 128, accumulationTolerance(growing, spaceType));
            }
        }
    }

    /**
     * Realistic values are not exactly representable. The exact tests above carry the parity proof at
     * these same dimensions, so this one only establishes that real inputs stay within accumulation
     * error rather than discriminating a structural defect on its own.
     */
    @Test
    public void testMatchesScalarForRealisticQueries() {
        for (int dimension : new int[] { 8, 13, 64, 100, 768, 772, 1536 }) {
            for (SpaceType spaceType : SUPPORTED) {
                final Random random = new Random(seedFor(dimension, spaceType));
                final float[] query = new float[dimension];
                for (int i = 0; i < dimension; i++) {
                    query[i] = (float) random.nextGaussian();
                }
                assertParity(dimension, spaceType, query, 128, accumulationTolerance(query, spaceType));
            }
        }
    }

    /** Cosine reaches ADC as an inner product, so it must produce the inner product distance. */
    @Test
    public void testCosineMatchesInnerProductReference() {
        final int dimension = 768;
        final float[] query = dyadicQuery(7, dimension);
        final byte[][] codes = randomCodes(new Random(7), dimension, 64);

        final AdcQueryTable cosine = AdcQueryTable.create(query, SpaceType.COSINESIMIL);
        for (byte[] code : codes) {
            // Against the reference, not against a sibling table, so a wrong correction cannot pass.
            assertEquals(KNNScoringUtil.innerProductADC(query, code), cosine.distance(code), 0.0f);
        }
    }

    @Test
    public void testUnsupportedSpaceTypeThrows() {
        final float[] query = dyadicQuery(3, 64);
        for (SpaceType spaceType : new SpaceType[] { SpaceType.HAMMING, SpaceType.L1, SpaceType.LINF }) {
            final IllegalArgumentException exception = expectThrows(
                IllegalArgumentException.class,
                () -> AdcQueryTable.create(query, spaceType)
            );
            assertTrue(exception.getMessage(), exception.getMessage().contains("no ADC distance computation"));
        }
    }

    /**
     * The scalar tail is a complete fallback, so shrinking the grouped coverage stays correct and only
     * costs throughput. Nothing else in this file would notice, hence the direct assertion.
     */
    @Test
    public void testTableCoversEveryWholeGroup() {
        for (int dimension : dimensions()) {
            for (SpaceType spaceType : SUPPORTED) {
                final AdcQueryTable table = AdcQueryTable.create(dyadicQuery(1, dimension), spaceType);
                assertEquals("dimension=" + dimension, dimension / 4, table.groups);
            }
        }
    }

    /**
     * A dimension that is zero across the training sample makes the ADC transform divide by zero, so a
     * non-finite coordinate is reachable. Its bit is then never set, and skipping unset bits keeps the
     * score finite where multiplying by them does not.
     */
    @Test
    public void testNonFiniteCoordinateOnUnsetBit() {
        final float[] query = new float[8];
        Arrays.fill(query, 0.5f);
        query[3] = Float.POSITIVE_INFINITY;
        final byte[] code = { (byte) 0b1110_1111 };

        assertTrue(Float.isNaN(KNNScoringUtil.innerProductADC(query, code)));
        assertEquals(3.5f, AdcQueryTable.create(query, SpaceType.INNER_PRODUCT).distance(code), 0.0f);

        // L2 folds every dimension into the correction, so both forms stay infinite.
        assertEquals(Float.POSITIVE_INFINITY, KNNScoringUtil.l2SquaredADC(query, code), 0.0f);
        assertEquals(Float.POSITIVE_INFINITY, AdcQueryTable.create(query, SpaceType.L2).distance(code), 0.0f);
    }

    /** Bits past the last dimension are padding and must not reach the distance. */
    @Test
    public void testPaddingBitsAreIgnored() {
        for (int dimension : new int[] { 1, 5, 12, 13, 20, 100, 772 }) {
            final int codeLength = (dimension + 7) / 8;
            final float[] query = new float[dimension];
            Arrays.fill(query, 1.0f);
            final AdcQueryTable table = AdcQueryTable.create(query, SpaceType.INNER_PRODUCT);
            for (int padding = dimension; padding < codeLength * 8; padding++) {
                final byte[] code = new byte[codeLength];
                code[padding >> 3] |= (byte) (1 << (7 - (padding & 7)));
                assertEquals("dimension=" + dimension + " padding bit=" + padding, 0.0f, table.distance(code), 0.0f);
            }
        }
    }

    private void assertParity(
        final int dimension,
        final SpaceType spaceType,
        final float[] query,
        final int documentCount,
        final float tolerance
    ) {
        final byte[][] codes = randomCodes(new Random(seedFor(dimension, spaceType)), dimension, documentCount);
        final AdcQueryTable table = AdcQueryTable.create(query, spaceType);
        for (int i = 0; i < codes.length; i++) {
            final float expected = spaceType == SpaceType.L2
                ? KNNScoringUtil.l2SquaredADC(query, codes[i])
                : KNNScoringUtil.innerProductADC(query, codes[i]);
            assertEquals(
                "dimension=" + dimension + " spaceType=" + spaceType + " document=" + i,
                expected,
                table.distance(codes[i]),
                tolerance
            );
        }
    }

    /**
     * Terms times the largest running sum times one ulp. L2 sums {@code (bit - q[i])^2}, so its partial
     * sums reach {@code n + 2|q| + ||q||^2}; inner product terms cancel, so {@code |q|} bounds it.
     */
    private static float accumulationTolerance(final float[] query, final SpaceType spaceType) {
        float squaredNorm = 0;
        float absoluteSum = 0;
        for (float value : query) {
            squaredNorm += value * value;
            absoluteSum += Math.abs(value);
        }
        final float largestRunningSum = spaceType == SpaceType.L2 ? query.length + 2 * absoluteSum + squaredNorm : absoluteSum;
        return query.length * largestRunningSum * Math.ulp(1.0f);
    }

    private static long seedFor(final int dimension, final SpaceType spaceType) {
        return dimension * 1_000_003L + spaceType.ordinal();
    }

    private static float[] dyadicQuery(final long seed, final int dimension) {
        final Random random = new Random(seed);
        final float[] query = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            query[i] = DYADIC[random.nextInt(DYADIC.length)];
        }
        return query;
    }

    /** Random padding bits and negative bytes, where production zero-fills; strictly harder. */
    private static byte[][] randomCodes(final Random random, final int dimension, final int count) {
        final byte[][] codes = new byte[count][(dimension + 7) / 8];
        for (byte[] code : codes) {
            random.nextBytes(code);
        }
        return codes;
    }
}
