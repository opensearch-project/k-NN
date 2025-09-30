/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.InaccessibleObjectException;

/**
 * For JDK 22 and later, the Foreign Function  &amp; Memory (FFM) API is part of the standard, so we no longer need to rely on reflection.
 * This class relies on reflection to extract {@code MemorySegment[]} from {@code IndexInput}.
 * And after acquired {@code MemorySegment[]}, it will use it right away to collect address and mapped size.
 */
@Log4j2
public final class MemorySegmentAddressExtractorJDK22 implements MemorySegmentAddressExtractor {
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
    public long[] extractAddressAndSize(final IndexInput indexInput) {
        try {
            // MMapDirectory in Lucene will return MemorySegmentIndexInput$SingleSegmentImpl or .$MultiSegmentImpl.
            // Thus, get the super class (e.g. MemorySegmentIndexInput) to acquire `MemorySegment[] segments`.
            final Field f = indexInput.getClass().getSuperclass().getDeclaredField("segments");
            f.setAccessible(true);
            final MemorySegment[] segments = (MemorySegment[]) f.get(indexInput);
            if (segments == null || segments.length == 0) {
                return null;
            }
            final long[] addressAndSize = new long[2 * segments.length];
            for (int segmentIndex = 0; segmentIndex < segments.length; ++segmentIndex) {
                final int addressIndex = 2 * segmentIndex;
                final int sizeIndex = 2 * segmentIndex + 1;
                final MemorySegment segment = segments[segmentIndex];
                if (segment == null) {
                    // Memory segments does not have complete mapped regions.
                    return null;
                }
                addressAndSize[addressIndex] = segments[segmentIndex].address();
                addressAndSize[sizeIndex] = segments[segmentIndex].byteSize();
            }
            return addressAndSize;
        } catch (NoSuchFieldException | IllegalAccessException | InaccessibleObjectException e) {
            // Ignore
            log.warn("Failed to extract MemorySegment[] from IndexInput, error message={}", e.getMessage());
        }
        return null;
    }
}
