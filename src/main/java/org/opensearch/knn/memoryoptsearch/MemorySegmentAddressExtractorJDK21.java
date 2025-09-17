/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.store.IndexInput;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;

/**
 * Since {@code MemorySegment} is not made officially in JDK21, we first extract {@code MemorySegment[]} from {@link IndexInput},
 * and have it as Object then use reflection to collect address and size info.
 */
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
    public long[] extractAddressAndSize(final IndexInput indexInput) {
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
            if (objSegments != null && !objSegments.getClass().isArray()) {
                // It's not MemorySegment[]
                return null;
            }
            final int numSegments = Array.getLength(objSegments);
            final long[] addressAndSize = new long[2 * numSegments];
            for (int i = 0; i < numSegments; i += 2) {
                final Object memorySegment = Array.get(objSegments, i);
                addressAndSize[i] = (long) ADDRESS_METHOD.invoke(memorySegment);
                addressAndSize[i + 1] = (long) BYTE_SIZE_METHOD.invoke(memorySegment);
            }

            return addressAndSize;
        } catch (NoSuchFieldException | IllegalAccessException | IllegalArgumentException e) {
            // Ignore
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        }
        return null;
    }
}
