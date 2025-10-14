/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.extern.log4j.Log4j2;

import java.lang.foreign.MemorySegment;

/**
 * For JDK 22 and later, the Foreign Function  &amp; Memory (FFM) API is part of the standard, so we no longer need to rely on reflection.
 * This class relies on reflection to extract {@code MemorySegment[]} from {@code IndexInput}.
 * And after acquired {@code MemorySegment[]}, it will use it right away to collect address and mapped size.
 */
@Log4j2
public final class MemorySegmentAddressExtractorJDK22 extends AbstractMemorySegmentAddressExtractor {
    @Override
    protected long getChunkSizeFromMemorySegment(Object memorySegment) {
        return ((MemorySegment) memorySegment).byteSize();
    }

    @Override
    protected long getAddressFromMemorySegment(Object memorySegment) {
        return ((MemorySegment) memorySegment).address();
    }
}
