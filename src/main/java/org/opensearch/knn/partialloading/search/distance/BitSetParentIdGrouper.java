/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search.distance;

import org.opensearch.knn.partialloading.search.DocIdGrouper;

public class BitSetParentIdGrouper implements DocIdGrouper {
    private long[] bits;

    public static BitSetParentIdGrouper createGrouper(int[] parentIds) {
        // Got an empty parent ids, returning null.
        if (parentIds == null || parentIds.length == 0) {
            return null;
        }

        // Find a max value
        int maxValue = Integer.MIN_VALUE;
        for (int parentId : parentIds) {
            if (parentId > maxValue) {
                maxValue = parentId;
            }
        }

        // Make bits whose length is always greater than 0.
        final int numBits = maxValue + 1;
        final int numBlocks = (numBits / Long.SIZE) + 1;
        final long[] bits = new long[numBlocks];

        // Covert int[] to bitset.
        // Ex: 15 -> located at the first uin64 block -> 0b...10000_00000_00000
        // Ex: 65 -> located at the second uint64 block (e.g. 1th = 65 / 64 when 0 based) -> 0b...10
        for (final int parentId : parentIds) {
            final int blockIndex = parentId / Long.SIZE;
            bits[blockIndex] |= 1L << (parentId % Long.SIZE);
        }

        return new BitSetParentIdGrouper(bits);
    }

    private BitSetParentIdGrouper(long[] bits) {
        this.bits = bits;
    }

    @Override
    public int getGroupId(int childDocId) {
        int blockIndex = childDocId / Long.SIZE;
        // Ex: childDocId = 66, parentId = 68
        // uint64 block -> 0b...1(68) 0(67) 0(66) 0(65) 0(64)
        // then we shift this block to truncate values less than 66
        // resulting block -> 0b...1(68) 0(67) 0(66)
        // lastly, we can get how much the parentId is far away from childDocId via Long.numberOfTrailingZeros, which is 2.
        // e.g. 68 = 66 + 2(=Long.numberOfTrailingZeros)
        final long shiftedBlock = bits[blockIndex] >> (childDocId % Long.SIZE);

        if (shiftedBlock != 0) {
            return childDocId + Long.numberOfTrailingZeros(shiftedBlock);
        }

        // Parent id is in the next block.
        // Ex: [0b00000000.., 0b...000010000]
        //          ^- child doc id    ^- this is the parent id
        while (++blockIndex < bits.length) {
            final long block = bits[blockIndex];
            if (block != 0) {
                return (blockIndex * Long.SIZE) + Long.numberOfTrailingZeros(block);
            }
        }

        // Keep the behavior consistent with FAISS where it returns the max value whenever failed to find parent id.
        return Integer.MAX_VALUE;
    }
}
