/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import org.apache.logging.log4j.Logger;
import org.junit.Test;
import org.opensearch.common.StopWatch;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class StopWatchUtilsTests {
    @Test
    public void shouldNotLoggingWhenNoDebug() {
        // Setting mock
        final Logger log = mock(Logger.class);
        when(log.isDebugEnabled()).thenReturn(false);

        // We should get null StopWatch if debug is disabled.
        final StopWatch stopWatch = StopWatchUtils.startStopWatch(log);
        assertNull(stopWatch);

        // It's safe to call stopping
        StopWatchUtils.stopStopWatchAndLog(log, stopWatch, "PrefixMessage", 0, "SegmentName", "FieldName");

        // Logger never called.
        verify(log, never()).debug(anyString());
    }

    @Test
    public void shouldLoggingWhenDebug() {
        // Setting mock
        final Logger log = mock(Logger.class);
        when(log.isDebugEnabled()).thenReturn(true);

        // We should get log when debugging is enabled
        final StopWatch stopWatch = StopWatchUtils.startStopWatch(log);
        assertNotNull(stopWatch);

        // Debug logging should be called
        StopWatchUtils.stopStopWatchAndLog(log, stopWatch, "PrefixMessage", 0, "SegmentName", "FieldName");
        verify(log, times(1)).debug(anyString(), eq(0), eq("SegmentName"), eq("FieldName"), anyLong());
    }
}
