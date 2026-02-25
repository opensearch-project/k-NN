/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.InaccessibleObjectException;

/**
 * From {@link IndexInput}, this class tries to extract mapped address with size.
 * When {@link IndexInput} created from {@link org.apache.lucene.store.MMapDirectory} was given, then we can extract
 * mapped pointer from it for faster computation.
 */
@Log4j2
public abstract class AbstractMemorySegmentAddressExtractor implements MemorySegmentAddressExtractor {
    /**
     * Try to extract {@code MemorySegment[]} from given input stream, and return address and size info of them.
     *
     * @param indexInput : Input stream
     * @param baseOffset : The offset used to determine which chunks to include.
     *                     Only chunks whose end offset is greater than baseOffset are collected, while those ending before baseOffset are
     *                     excluded.
     * @param requestSize : The total number of bytes (or chunk size) to extract starting from {@code baseOffset}.
     *               This value determines how much data to include in the returned segments.
     * @return null if it fails to extract mapped pointer otherwise it will return an array having address and size.
     *         Ex: address_i = array[i], size_i = array[i + 1].
     */
    @Override
    public long[] extractAddressAndSize(IndexInput indexInput, long baseOffset, long requestSize) {
        try {
            return doExtractAddressAndSize(indexInput, baseOffset, requestSize);
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            // Unexpected exception is thrown. Logging and return null, we should not halt the search, logging error and let it fallback to
            // default logic.
            log.error("Unexpected exception was thrown from address extraction", e);
        }

        return null;
    }

    private long[] doExtractAddressAndSize(IndexInput indexInput, long baseOffset, long requestSize) {
        // We're expecting this to be MemorySegment[]
        final Object objSegments = getMemorySegments(indexInput);
        if (objSegments == null) {
            // It's not MemorySegment[]
            return null;
        }

        final int numSegments = Array.getLength(objSegments);
        final long[] addressAndSize = new long[2 * numSegments];
        int addressIndex = 0, sizeIndex = 1;
        long startOffset = 0;
        long totalSize = 0;

        for (int segmentIndex = 0; segmentIndex < numSegments; segmentIndex++) {
            final Object memorySegment = Array.get(objSegments, segmentIndex);
            if (memorySegment == null) {
                // Memory segments does not have complete mapped regions.
                log.warn(
                    "Memory segment at " + segmentIndex + " is null, which is unexpected. The number of MemorySegment was" + numSegments
                );
                return null;
            }
            long address = getAddressFromMemorySegment(memorySegment);
            long chunkSize = getChunkSizeFromMemorySegment(memorySegment);

            final long originalChunkSize = chunkSize;
            final long endOffsetExclusive = startOffset + chunkSize;
            if (endOffsetExclusive > baseOffset) {
                // If this chunk contains `baseOffset`, then force its address value to be `baseOffset`
                if (startOffset < baseOffset) {
                    chunkSize = endOffsetExclusive - baseOffset;
                    address += baseOffset - startOffset;
                }

                // Collect only chunk that overlap with baseOffset or placed after baseOffset
                addressAndSize[addressIndex] = address;
                addressAndSize[sizeIndex] = chunkSize;
                addressIndex += 2;

                // Check to see if this is the last chunk
                totalSize += chunkSize;
                if (totalSize >= requestSize) {
                    // <----------------------------------------> Total size accumulated
                    // <-----------------------------><--------->
                    // Request size ------^ ^--------- Total size - requestSize
                    // This is the last chunk, adjusting size is required.
                    addressAndSize[sizeIndex] -= totalSize - requestSize;
                    totalSize = requestSize;
                    break;
                }

                sizeIndex += 2;
            }

            startOffset += originalChunkSize;
        }

        if (requestSize > totalSize) {
            throw new IllegalArgumentException(
                "Requested size (" + requestSize + " bytes) exceeds available memory chunk size (" + totalSize + " bytes)."
            );
        }

        if (addressIndex != addressAndSize.length) {
            // There was a chunk that excluded, shrink it
            final long[] newAddressAndSize = new long[addressIndex];
            System.arraycopy(addressAndSize, 0, newAddressAndSize, 0, addressIndex);
            return newAddressAndSize;
        }

        return addressAndSize;
    }

    protected abstract long getChunkSizeFromMemorySegment(Object memorySegment);

    protected abstract long getAddressFromMemorySegment(Object memorySegment);

    protected Object getMemorySegments(final IndexInput indexInput) {
        try {
            // MMapDirectory in Lucene will return MemorySegmentIndexInput$SingleSegmentImpl or .$MultiSegmentImpl.
            // Thus, get the super class (e.g. MemorySegmentIndexInput) to acquire `MemorySegment[] segments`.
            final Field f = indexInput.getClass().getSuperclass().getDeclaredField("segments");
            f.setAccessible(true);
            // We're expecting this to be MemorySegment[]
            final Object objSegments = f.get(indexInput);
            if (objSegments == null || !objSegments.getClass().isArray()) {
                // It's not MemorySegment[]
                return null;
            }
            return objSegments;
        } catch (IllegalAccessException | InaccessibleObjectException e) {
            // MemorySegmentIndexInput was provided, but encountered an unexpected exception.
            log.warn(
                "Failed to extract MemorySegment[] from IndexInput=" + indexInput.getClass().getSimpleName() + " , error message={}",
                e.getMessage()
            );
        } catch (NoSuchFieldException e) {
            // Ignore, this is not MemorySegmentIndexInput.
        }

        return null;
    }
}
