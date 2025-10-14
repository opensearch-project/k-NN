/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;

import java.lang.reflect.InvocationTargetException;

/**
 * Since {@code MemorySegment} is not made officially in JDK21, we first extract {@code MemorySegment[]} from {@link IndexInput},
 * and have it as Object then use reflection to collect address and size info.
 */
@Log4j2
public final class MemorySegmentAddressExtractorJDK21 extends AbstractMemorySegmentAddressExtractor {
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

    @Override
    protected long getChunkSizeFromMemorySegment(final Object memorySegment) {
        try {
            return (long) BYTE_SIZE_METHOD.invoke(memorySegment);
        } catch (InvocationTargetException | IllegalAccessException e) {
            // MemorySegmentIndexInput was provided, but encountered an unexpected exception.
            log.warn("Failed to get chunk size from MemorySegment, error message={}", e.getMessage());
            throw new RuntimeException(e);
        }
    }

    @Override
    protected long getAddressFromMemorySegment(final Object memorySegment) {
        try {
            return (long) ADDRESS_METHOD.invoke(memorySegment);
        } catch (InvocationTargetException | IllegalAccessException e) {
            // MemorySegmentIndexInput was provided, but encountered an unexpected exception.
            log.warn("Failed to get address from MemorySegment, error message={}", e.getMessage());
            throw new RuntimeException(e);
        }
    }

    @Override
    protected Object getMemorySegments(final IndexInput indexInput) {
        if (ADDRESS_METHOD == null || BYTE_SIZE_METHOD == null) {
            // Runtime JDK does not support MemorySegment.
            return null;
        }

        return super.getMemorySegments(indexInput);
    }
}
