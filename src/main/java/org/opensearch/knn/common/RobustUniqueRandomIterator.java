/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import java.util.concurrent.ThreadLocalRandom;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * An iterator that provides unique random integers in the range [0, maxValExclusive)
 * using O(1) memory and O(1) complexity (amortized).
 *
 * <h2>Theory of Operation</h2>
 * This class utilizes a <b>Linear Congruential Generator (LCG)</b> configured to produce
 * a "Full Period" (or Full Cycle). A standard LCG follows the recurrence relation.
 * Formula: X_{n+1} = (a * X_n + c) mod M
 *
 * <h3>1. The Hull-Dobell Theorem</h3>
 * To ensure the generator visits every single number in the range [0, M) exactly once
 * before repeating (uniqueness), the parameters must satisfy:
 * <ul>
 * <li>c and M are relatively prime (satisfied here as c is odd and M is a power of 2).</li>
 * <li>a - 1 is divisible by all prime factors of M.</li>
 * <li>If M is a multiple of 4, a - 1 must be a multiple of 4 (satisfied by a = 4k + 1).</li>
 * </ul>
 *
 * <h3>2. The "Sandbox" &amp; Rejection Sampling</h3>
 * LCGs are most efficient when M is a power of two, as the modulo operation becomes a
 * bitwise {@code & (M - 1)}. However, the user's requested {@code maxValExclusive} might not
 * be a power of two.
 * <p>
 * This class solves this by finding the smallest power of two (M) that is greater than or
 * equal to the requested range. When the LCG generates a value X such that
 * {@code maxValExclusive <= X < M}, the value is "rejected," and the next value in the
 * sequence is generated. Because M &lt; 2 * maxValExclusive, the expected number of
 * iterations is less than 2, preserving O(1) amortized time.
 * </p>
 */
public final class RobustUniqueRandomIterator {
    /** The user's requested max value (exclusive). */
    private final int maxValExclusive;

    /** The internal power-of-two range used for the LCG math. */
    private final long M;

    /** The current state of the LCG (the last generated value in the full cycle). */
    private long current;

    /** Tracks how many elements have been successfully returned to the user. */
    private long populated = 0;

    /** The total number of elements the user wants to pick. */
    private final int numPopulate;

    /**
     * @param maxValExclusive The upper bound (exclusive) of the random numbers.
     * @param numPopulate     How many unique numbers to pick.
     * @throws IllegalArgumentException if numPopulate > maxValExclusive.
     */
    public RobustUniqueRandomIterator(int maxValExclusive, int numPopulate) {
        if (maxValExclusive <= 0) {
            throw new IllegalArgumentException(String.format("maxValExclusive[%d] must be positive", maxValExclusive));
        }
        if (numPopulate < 0) {
            throw new IllegalArgumentException(String.format("numPopulate[%d] must be non-negative", numPopulate));
        }
        if (numPopulate > maxValExclusive) {
            throw new IllegalArgumentException(String.format("numPopulate[%d] > maxValExclusive[%d]", numPopulate, maxValExclusive));
        }

        this.maxValExclusive = maxValExclusive;
        this.numPopulate = numPopulate;

        // Calculate M: The smallest power of 2 >= maxValExclusive.
        // If maxValExclusive is 77,777:
        // (77,776) in binary is 00010010111111010000...
        // highestOneBit returns 65,536. Shift left 1 gives 131,072.
        if (maxValExclusive <= 1) {
            this.M = 1;
        } else {
            this.M = Long.highestOneBit(maxValExclusive - 1) << 1;
        }

        // Seed the LCG with a random starting point within the full cycle [0, M).
        this.current = ThreadLocalRandom.current().nextLong(M);
    }

    /**
     * Generates the next unique random element.
     *
     * @return A unique random integer in range [0, maxValExclusive),
     * or NO_MORE_DOCS if no elements remain.
     */
    public int next() {
        if (populated >= numPopulate) {
            return NO_MORE_DOCS;
        }

        // Optimization: M is a power of 2, so (X % M) is equivalent to (X & (M - 1)).
        final long mask = M - 1;

        // Multiplier (a): Must be (4k + 1) for a power-of-two M to satisfy Hull-Dobell.
        // Using a multiplier from Knuth's suggestions for LCGs.
        final long a = 1664525L * 4 + 1;

        // Increment (c): Must be odd to be relatively prime to the power-of-two M.
        final long c = 1013904223L | 1;

        // Rejection Sampling Loop:
        // We traverse the 'Sandbox' [0, M). If a value is outside the user's
        // range [0, maxValExclusive), we skip it and move to the next in the cycle.
        long nextVal;
        do {
            // Faithfully follow the formula : X_{n+1} = (a * X_n + c) mod M
            current = (a * current + c) & mask;
            nextVal = current;
        } while (nextVal >= maxValExclusive);

        ++populated;
        return (int) nextVal;
    }
}
