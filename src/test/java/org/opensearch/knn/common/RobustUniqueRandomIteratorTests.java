/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.opensearch.knn.KNNTestCase;

import java.util.HashSet;
import java.util.Set;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class RobustUniqueRandomIteratorTests extends KNNTestCase {

    // --- Constructor validation edge cases ---

    public void testConstructorThrowsWhenMaxValExclusiveIsZero() {
        expectThrows(IllegalArgumentException.class, () -> new RobustUniqueRandomIterator(0, 0));
    }

    public void testConstructorThrowsWhenMaxValExclusiveIsNegative() {
        expectThrows(IllegalArgumentException.class, () -> new RobustUniqueRandomIterator(-1, 0));
    }

    public void testConstructorThrowsWhenNumPopulateIsNegative() {
        expectThrows(IllegalArgumentException.class, () -> new RobustUniqueRandomIterator(10, -1));
    }

    public void testConstructorThrowsWhenNumPopulateExceedsMaxVal() {
        expectThrows(IllegalArgumentException.class, () -> new RobustUniqueRandomIterator(5, 6));
    }

    // --- Zero populate ---

    public void testZeroPopulateReturnsNoMoreDocsImmediately() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(10, 0);
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- maxValExclusive = 1 (single element) ---

    public void testSingleElementRange() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(1, 1);
        assertEquals(0, iter.next());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Power-of-two range (no rejection needed) ---

    public void testPowerOfTwoRange() {
        int maxVal = 8;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, maxVal);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < maxVal; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Non-power-of-two range (rejection sampling exercised) ---

    public void testNonPowerOfTwoRange() {
        int maxVal = 77_777;
        int numPopulate = 1000;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, numPopulate);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < numPopulate; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Full exhaustion: numPopulate == maxValExclusive ---

    public void testFullExhaustionSmallRange() {
        int maxVal = 100;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, maxVal);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < maxVal; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals("Should have all values", maxVal, seen.size());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Partial sampling: numPopulate < maxValExclusive ---

    public void testPartialSampling() {
        int maxVal = 500;
        int numPopulate = 50;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, numPopulate);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < numPopulate; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(numPopulate, seen.size());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- maxValExclusive = 2 (smallest non-trivial range) ---

    public void testRangeOfTwo() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(2, 2);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 2; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < 2);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- numPopulate == maxValExclusive boundary ---

    public void testNumPopulateEqualsMaxVal() {
        int maxVal = 10;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, maxVal);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < maxVal; i++) {
            int val = iter.next();
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(maxVal, seen.size());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Multiple calls past exhaustion still return NO_MORE_DOCS ---

    public void testRepeatedCallsAfterExhaustion() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(5, 2);
        iter.next();
        iter.next();
        assertEquals(NO_MORE_DOCS, iter.next());
        assertEquals(NO_MORE_DOCS, iter.next());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Large range uniqueness stress test ---

    public void testLargeRangeUniqueness() {
        int maxVal = 100_000;
        int numPopulate = 10_000;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, numPopulate);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < numPopulate; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate at iteration " + i + ": " + val, seen.add(val));
        }
        assertEquals(numPopulate, seen.size());
    }

    // --- Boundary: maxValExclusive = 3 (non-power-of-two, small) ---

    public void testSmallNonPowerOfTwo() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(3, 3);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 3; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < 3);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(3, seen.size());
        assertEquals(NO_MORE_DOCS, iter.next());
    }
}
