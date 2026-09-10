/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import lombok.experimental.UtilityClass;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.opensearch.common.Nullable;

/**
 * Helpers for working with a {@link LeafReader} that may not be backed by a segment.
 */
@UtilityClass
public class LeafReaderUtil {

    /**
     * Returns the {@link SegmentReader} backing the given reader, or {@code null} when the reader is not
     * backed by a segment. Mirrors the unwrapping {@code Lucene#segmentReader} performs, without its hard
     * failure.
     *
     * @param reader the leaf reader to unwrap
     * @return the backing segment reader, or {@code null} if there is none
     */
    @Nullable
    public static SegmentReader tryGetSegmentReader(final LeafReader reader) {
        if (reader instanceof SegmentReader segmentReader) {
            return segmentReader;
        }
        if (reader instanceof FilterLeafReader filterLeafReader) {
            return tryGetSegmentReader(FilterLeafReader.unwrap(filterLeafReader));
        }
        if (reader instanceof FilterCodecReader filterCodecReader) {
            return tryGetSegmentReader(FilterCodecReader.unwrap(filterCodecReader));
        }
        return null;
    }

    /**
     * Names a leaf for logging, using the segment name when there is one.
     *
     * @param reader the leaf reader to name
     * @return the segment name, or the reader's class name when there is no segment
     */
    public static String leafReaderName(final LeafReader reader) {
        final SegmentReader segmentReader = tryGetSegmentReader(reader);
        return segmentReader == null ? reader.getClass().getSimpleName() : segmentReader.getSegmentName();
    }
}
