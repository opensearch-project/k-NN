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
import java.lang.reflect.InvocationTargetException;

/**
 * Since {@code MemorySegment} is not made officially in JDK21, we first extract {@code MemorySegment[]} from {@link IndexInput},
 * and have it as Object then use reflection to collect address and size info.
 */
@Log4j2
public final class MemorySegmentAddressExtractorJDK21 implements MemorySegmentAddressExtractor {
    private static final java.lang.reflect.Method ADDRESS_METHOD;
    private static final java.lang.reflect.Method BYTE_SIZE_METHOD;

    // Get the handle when loading once.
    static {
        java.lang.reflect.Method addressMethod = null;
        java.lang.reflect.Method byteSizeMethod = null;
        try {
            Class<?> clazz = Class.forName("java.lang.foreign.MemorySegment");
            addressMethod = clazz.getMethod("address");
            byteSizeMethod = clazz.getMethod("byteSize");
        } catch (ClassNotFoundException | NoSuchMethodException e) {
            // Running on JDK where MemorySegment/address() doesn't exist
        }
        ADDRESS_METHOD = addressMethod;
        BYTE_SIZE_METHOD = byteSizeMethod;
    }

    /**
     * Extracts address and size info of {@code MemorySegment[]} from the given {@code indexInput}.
     * <p>
     * When using {@code MMapDirectory}, the {@code indexInput} may be an instance of
     * {@code MemorySegmentIndexInput$SingleSegmentImpl} or {@code MemorySegmentIndexInput$MultiSegmentImpl}.
     * These classes wrap mapped pointers in {@code MemorySegment} objects stored in a field named {@code segments}.
     * This method detects the {@code segments} field, extracts its value, and returns it.
     * <p>
     * If the corresponding `segments` cannot be found, this method simply returns {@code null}.
     * In that case, the search logic falls back to the default scorer, which loads vectors
     * into the JVM heap and performs distance calculations there.
     * <p>
     *
     * @param indexInput the input from which to extract memory segments
     * @return an array of address and size info extracted from the input, or null if it's not found.
     *         ex: address_i = array[i], size_i = array[i + 1]
     */
    @Override
    public long[] extractAddressAndSize(final IndexInput indexInput, long baseOffset) {
        if (ADDRESS_METHOD == null || BYTE_SIZE_METHOD == null) {
            // Runtime JDK does not support MemorySegment.
            return null;
        }

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
            final int numSegments = Array.getLength(objSegments);
            final long[] addressAndSize = new long[2 * numSegments];
            int addressIndex = 0, sizeIndex = 1;
            long startOffset = 0;
            for (int segmentIndex = 0; segmentIndex < numSegments; segmentIndex++) {
                final Object memorySegment = Array.get(objSegments, segmentIndex);
                if (memorySegment == null) {
                    // Memory segments does not have complete mapped regions.
                    log.warn("Memory segment at " + segmentIndex + " is null, which is unexpected. " + "The number of MemorySegment was"
                             + numSegments);
                    return null;
                }

                long address = (long) ADDRESS_METHOD.invoke(memorySegment);
                long chunkSize = (long) BYTE_SIZE_METHOD.invoke(memorySegment);
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
                    addressIndex += 2;
                    addressAndSize[sizeIndex] = chunkSize;
                    sizeIndex += 2;
                }
                startOffset += originalChunkSize;
            }

            if (addressIndex != addressAndSize.length) {
                // There was a chunk that excluded, shrink it
                long[] newAddressAndSize = new long[addressIndex];
                System.arraycopy(addressAndSize, 0, newAddressAndSize, 0, addressIndex);
                return newAddressAndSize;
            }

            return addressAndSize;
        } catch (IllegalAccessException | IllegalArgumentException | InaccessibleObjectException | InvocationTargetException e) {
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
