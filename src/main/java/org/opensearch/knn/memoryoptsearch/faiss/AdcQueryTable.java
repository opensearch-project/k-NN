/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.opensearch.knn.index.SpaceType;

import java.util.Arrays;

import java.util.Locale;

/**
 * Per-query state for ADC (asymmetric distance computation) scoring, where a full precision float
 * query is scored against 1-bit quantized document vectors.
 *
 * <p>A set bit's contribution depends only on the query, so this sums the contribution of all
 * sixteen patterns each group of four bits can hold, once per query. Scoring a document is then two
 * table reads per code byte instead of a shift, mask and multiply per dimension.
 *
 * <p>Group size eight halves the reads but costs eight times the build and eight times the memory,
 * which it only repays past roughly eight hundred documents in one segment, by at most 1.7x. The
 * document count is unknown when the table is built, so this stays at four bits, which is ahead from
 * a handful of documents upwards.
 *
 * <p>Terms are summed in a different order than the per-dimension loop, so results differ in the last
 * bits of the float for queries whose values are not exactly representable; grouping is a partial
 * pairwise summation, so this is closer to the exact value more often.
 *
 * <p>Inner product diverges on a non-finite coordinate: the per-dimension loop multiplies it by an
 * unset bit and returns NaN, while this skips unset bits and stays finite. L2 returns infinity either
 * way, unless the squared norm overflows, where this returns NaN.
 *
 * <p>Immutable. One is built per query per segment, since the ADC transform is segment-level.
 */
final class AdcQueryTable {

    private static final int GROUP_BITS = 4;
    private static final int PATTERNS_PER_GROUP = 1 << GROUP_BITS;

    /** Set-bit contributions for the dimensions past the last whole group. At most three. */
    private final float[] tailScores;

    /** Added to every distance. Non-zero only for L2, where an unset bit still contributes. */
    private final float correction;

    /** Every pattern of every whole group, {@code groups * PATTERNS_PER_GROUP} entries. */
    private final float[] table;

    /** Whole groups the table covers. Package-private so a test can pin the grouped coverage. */
    final int groups;

    private AdcQueryTable(final float[] coordinateScores, final float correction) {
        this.correction = correction;
        this.groups = coordinateScores.length / GROUP_BITS;
        this.table = buildTable(coordinateScores, groups);
        // Only the dimensions past the last whole group are read again, so the rest is not retained.
        this.tailScores = Arrays.copyOfRange(coordinateScores, groups * GROUP_BITS, coordinateScores.length);
    }

    /**
     * Builds the per-query table for the given space type.
     *
     * @param query     full precision query vector
     * @param spaceType L2, or inner product which cosine also resolves to
     * @return a table producing the same distance as the per-dimension loop
     * @throws IllegalArgumentException if the space type has no ADC distance
     */
    static AdcQueryTable create(final float[] query, final SpaceType spaceType) {
        return switch (spaceType) {
            case L2 -> forL2(query);
            // Cosine reaches ADC as an inner product over L2-normalized vectors.
            case INNER_PRODUCT, COSINESIMIL -> forInnerProduct(query);
            default -> throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Space type [%s] has no ADC distance computation", spaceType)
            );
        };
    }

    /**
     * An unset bit contributes q[i]^2 and a set bit (1 - q[i])^2. Rewriting the set case as
     * (1 - 2q[i]) + q[i]^2 separates a set-bit-only coefficient from a constant, which is what makes
     * the sum tabulatable.
     */
    private static AdcQueryTable forL2(final float[] query) {
        final float[] coordinateScores = new float[query.length];
        float correction = 0;
        for (int i = 0; i < query.length; i++) {
            coordinateScores[i] = 1 - 2 * query[i];
            correction += query[i] * query[i];
        }
        return new AdcQueryTable(coordinateScores, correction);
    }

    /** For inner product a set bit contributes q[i] and an unset bit contributes nothing. */
    private static AdcQueryTable forInnerProduct(final float[] query) {
        // Not copied: the constructor only reads it, and keeps a copy of the tail.
        return new AdcQueryTable(query, 0);
    }

    /**
     * Returns the ADC distance between the query and a packed 1-bit code. Agrees with
     * {@code KNNScoringUtil#l2SquaredADC} and {@code KNNScoringUtil#innerProductADC} to within float
     * accumulation order.
     *
     * @param code packed document code, most significant bit first
     * @return squared L2 distance, or raw inner product, depending on the space type
     */
    float distance(final byte[] code) {
        float sum0 = 0;
        float sum1 = 0;
        final int wholeBytes = groups >> 1;
        // Same reads as indexing by group number, but computing group * PATTERNS_PER_GROUP per lookup
        // measured about 1.5x slower.
        int base = 0;
        for (int i = 0; i < wholeBytes; i++, base += 2 * PATTERNS_PER_GROUP) {
            final int packed = code[i] & 0xFF;
            sum0 += table[base + (packed >>> GROUP_BITS)];
            sum1 += table[base + PATTERNS_PER_GROUP + (packed & 0xF)];
        }
        // An odd group count leaves a lone high nibble, which happens when dim mod 8 is four or more.
        // The loop above stops before that group, with base pointing at it.
        if ((groups & 1) != 0) {
            sum0 += table[base + ((code[wholeBytes] & 0xFF) >>> GROUP_BITS)];
        }
        return sum0 + sum1 + sumUngroupedTail(code) + correction;
    }

    /** Dimensions past the last whole group, scored one at a time. At most three of them. */
    private float sumUngroupedTail(final byte[] code) {
        float sum = 0;
        final int firstDimension = groups * GROUP_BITS;
        for (int j = 0; j < tailScores.length; j++) {
            final int i = firstDimension + j;
            if (((code[i >> 3] >> (7 - (i & 7))) & 1) != 0) {
                sum += tailScores[j];
            }
        }
        return sum;
    }

    /**
     * Builds the contribution of every bit pattern each group can hold. Each pattern is filled from a
     * shorter one already computed, so an entry costs one add.
     *
     * <p>Within a group, bit {@code i} counting from the least significant end belongs to the
     * dimension {@code GROUP_BITS - 1 - i} positions into the group, because codes are packed most
     * significant bit first. This has to agree with {@code BitPacker} on the write side.
     */
    private static float[] buildTable(final float[] coordinateScores, final int groups) {
        final float[] table = new float[groups * PATTERNS_PER_GROUP];
        for (int group = 0; group < groups; group++) {
            final int base = group * PATTERNS_PER_GROUP;
            for (int bit = 0; bit < GROUP_BITS; bit++) {
                final int patternBit = 1 << bit;
                final float contribution = coordinateScores[group * GROUP_BITS + (GROUP_BITS - 1 - bit)];
                for (int shorter = 0; shorter < patternBit; shorter++) {
                    table[base + (patternBit | shorter)] = table[base + shorter] + contribution;
                }
            }
        }
        return table;
    }
}
