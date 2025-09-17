/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.store.IndexInput;

public final class MemorySegmentAddressExtractorUtil {
    private static final MemorySegmentAddressExtractor INSTANCE;

    static {
        MemorySegmentAddressExtractor instance = null;
        try {
            // Try to load the JDK22-optimized class which will be packaged only for compile_target = 22+.
            Class<?> clazz = Class.forName("org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorJDK22");
            instance = (MemorySegmentAddressExtractor) clazz.getDeclaredConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            // Class not found: fall back to JDK21 version
            instance = new MemorySegmentAddressExtractorJDK21();
        } catch (Throwable t) {
            // Any other errors (constructor, reflection issues)
            throw new RuntimeException("Failed to initialize MemorySegmentAddressExtractor", t);
        }
        INSTANCE = instance;
    }

    public static long[] tryExtractAddressAndSize(final IndexInput indexInput) {
        return INSTANCE.extractAddressAndSize(indexInput);
    }
}
