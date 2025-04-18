/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.experimental.UtilityClass;
import org.apache.lucene.index.LeafReader;
import org.opensearch.knn.profiler.SegmentProfileKNNCollector;
import org.opensearch.knn.profiler.SegmentProfilerState;

import java.io.IOException;
import java.util.Locale;

/**
 * Utility class to get segment profiler state for a given field
 */
@UtilityClass
public class SegmentProfilerUtil {

    /**
     * Gets the segment profile state for a given field
     * @param leafReader The leaf reader to query
     * @param fieldName The field name to profile
     * @return The segment profiler state
     * @throws IOException If there's an error reading the segment
     */
    public static SegmentProfilerState getSegmentProfileState(final LeafReader leafReader, String fieldName) throws IOException {
        final SegmentProfileKNNCollector tempCollector = new SegmentProfileKNNCollector();
        leafReader.searchNearestVectors(fieldName, new float[0], tempCollector, null);
        if (tempCollector.getSegmentProfilerState() == null) {
            throw new IllegalStateException(String.format(Locale.ROOT, "No segment state found for field %s", fieldName));
        }
        return tempCollector.getSegmentProfilerState();
    }
}
