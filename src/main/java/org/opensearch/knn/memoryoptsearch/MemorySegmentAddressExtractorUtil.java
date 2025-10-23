/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;

/**
 * This class instantiates a {@code MemorySegment} address and size extractor when it is loaded.
 * It first attempts to load MemorySegmentAddressExtractorJDK22+; if that fails,
 * it falls back to {@link MemorySegmentAddressExtractorJDK21}.
 *
 * <p>
 * Note that MemorySegmentAddressExtractorJDK22+ is only included when the project
 * is compiled with JDK&nbsp;22+ compatibility.
 * </p>
 *
 * <p>
 * Both extractors use reflection to obtain the private {@code MemorySegment[]} field from
 * {@link IndexInput}. In particular, {@link org.apache.lucene.store.MMapDirectory} returns
 * an internal {@link IndexInput} implementation that stores {@code MemorySegment[]} in
 * a private field named {@code segment}. Reflection is used to access this field and
 * extract address and size information.
 * </p>
 *
 * <p>
 * For JDK&nbsp;21, {@link MemorySegmentAddressExtractorJDK21} also uses reflection to indirectly
 * invoke the {@code address()} and {@code byteSize()} methods of {@code MemorySegment}.
 * Since {@code MemorySegment} only became a standard API in JDK&nbsp;22+, reflection is the
 * only option unless the {@code --enable-preview} flag is explicitly provided.
 * </p>
 *
 * <pre>
 * +-----------------------------+        (reflection)        +--------------------------------------+
 * |          IndexInput         |  ----------------------->  |  MemorySegmentAddressExtractorJDK22  |
 * |  segments: MemorySegment[]  |                            |  Address[], Size[]                   |
 * +-----------------------------+                            +--------------------------------------+
 *
 * +-----------------------------+        (reflection)        +--------------------------------------+
 * |          IndexInput         |  ----------------------->  |  MemorySegmentAddressExtractorJDK21  |
 * |  segments: MemorySegment[]  |                     ^      |  Address[], Size[]                   |
 * +-----------------------------+                     |      +--------------------------------------+
 *                                                     |
 *             (reflection on MemorySegment methods)   +--> MemorySegment#address()
 *                                                         MemorySegment#byteSize()
 * </pre>
 */
@Log4j2
public final class MemorySegmentAddressExtractorUtil {
    private static final MemorySegmentAddressExtractor INSTANCE;

    static {
        MemorySegmentAddressExtractor instance;
        try {
            try {
                // Try to load the JDK22-optimized class which will be packaged only for compile_target = 22+.
                Class<?> clazz = Class.forName("org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorJDK22");
                instance = (AbstractMemorySegmentAddressExtractor) clazz.getDeclaredConstructor().newInstance();
                log.info("Loaded MemorySegmentAddressExtractorJDK22");
            } catch (ClassNotFoundException e) {
                log.warn("Failed to load MemorySegmentAddressExtractorJDK22, falling back to MemorySegmentAddressExtractorJDK21", e);
                // Class not found: fall back to JDK21 version
                instance = new MemorySegmentAddressExtractorJDK21();
            }
        } catch (Throwable t) {
            // Any other errors (constructor, reflection issues)
            log.error("Failed to instantiate MemorySegmentAddressExtractor", t);
            instance = (indexInput, baseOffset, requestSize) -> null;
        }
        INSTANCE = instance;
    }

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
    public static long[] tryExtractAddressAndSize(final IndexInput indexInput, final long baseOffset, final long requestSize) {
        return INSTANCE.extractAddressAndSize(indexInput, baseOffset, requestSize);
    }
}
