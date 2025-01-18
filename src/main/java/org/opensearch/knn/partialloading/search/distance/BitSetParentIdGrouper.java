/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search.distance;

import org.opensearch.knn.partialloading.search.DocIdGrouper;

/**
 * The mapping is implemented using a bitset to represent the relationship between a child ID and its parent ID.
 * We use "group ID" and "parent ID" interchangeably here, but generally, "group ID" is the more commonly used term in FAISS, while
 * "parent ID" is more specific to the OpenSearch index layout structure.
 * <p>
 * This grouper assumes the following:
 * <p>
 * 1. Child IDs are unique.
 * 2. Child IDs belonging to the same parent must be sequential, increasing by 1.
 * 3. The parent ID must be exactly 1 greater than its maximum child ID.
 *
 * <p>
 * Ex:
 * Parent id = 3, child ids = [0, 1, 2]
 * Parent id = 100, child ids = [96, 97, 98, 99] -> then its previous parent id is 95.
 */
public class BitSetParentIdGrouper implements DocIdGrouper {
    private final long[] bits;

    /**
     * Creates a grouper using the provided `parentIds`.
     * Please refer to the class-level comment for the assumptions that this implementation makes.
     *
     * @param parentIds A list of parent id.
     * @return {@link BitSetParentIdGrouper} returns a group id (e.g. parent id) per child id.
     */
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

    private BitSetParentIdGrouper(final long[] bits) {
        this.bits = bits;
    }

    /**
     * Gets the group ID for the given `childDocId`.
     * Note that the provided document ID should not be a physical vector ID within the FAISS index. Before calling this method, the
     * physical vector ID must be converted to a logical document ID and then passed.
     *
     * @param childDocId A child document id.
     * @return A group id that the given child document id belongs to.
     */
    @Override
    public int getGroupId(int childDocId) {
        int blockIndex = childDocId / Long.SIZE;
        // Ex: childDocId = 66, parentId = 68
        // uint64 block -> 0b...1(68) 0(67) 0(66) 0(65) 0(64)
        // then we shift this block to truncate values less than 66
        // resulting block -> 0b...1(68) 0(67) 0(66)
        // lastly, we can get how much the parentId is far away from childDocId via Long.numberOfTrailingZeros, which is 2.
        // e.g. parentDocId = 68 = 66 + 2(=Long.numberOfTrailingZeros)
        final long shiftedBlock = bits[blockIndex] >> (childDocId % Long.SIZE);

        if (shiftedBlock != 0) {
            // It's parent id is within the same uint64 block.
            return childDocId + Long.numberOfTrailingZeros(shiftedBlock);
        }

        // Parent id is somewhere in next blocks.
        // Ex: [0b00000000.., 0b...000010000, ...]
        //          ^- child doc id    ^- this is its parent id
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
