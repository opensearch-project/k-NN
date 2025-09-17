/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.store.IndexInput;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;

/**
 * For JDK 22 and later, the Foreign Function & Memory (FFM) API is part of the standard, so we no longer need to rely on reflection.
 * This class relies on reflection to extract {@code MemorySegment[]} from {@code IndexInput}.
 * And after acquired {@code MemorySegment[]}, it will use it right away to collect address and mapped size.
 */
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
    public long[] extractAddressAndSize(final IndexInput indexInput) {
        try {
            // MMapDirectory in Lucene will return MemorySegmentIndexInput$SingleSegmentImpl or .$MultiSegmentImpl.
            // Thus, get the super class (e.g. MemorySegmentIndexInput) to acquire `MemorySegment[] segments`.
            final Field f = indexInput.getClass().getSuperclass().getDeclaredField("segments");
            f.setAccessible(true);
            final MemorySegment[] segments = (MemorySegment[]) f.get(indexInput);
            if (segments == null) {
                return null;
            }
            final long[] addressAndSize = new long[2 * segments.length];
            for (int i = 0; i < segments.length; i += 2) {
                addressAndSize[i] = segments[i].address();
                addressAndSize[i + 1] = segments[i].byteSize();
            }
            return addressAndSize;
        } catch (NoSuchFieldException | IllegalAccessException ea) {
            // Ignore
        }
        return null;
    }
}
