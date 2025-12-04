/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import lombok.experimental.UtilityClass;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.Nullable;
import org.opensearch.common.StopWatch;

@UtilityClass
public final class StopWatchUtils {
    public static StopWatch startStopWatch(final Logger log) {
        if (log.isDebugEnabled()) {
            return new StopWatch().start();
        }
        return null;
    }

    public static void stopStopWatchAndLog(
        final Logger log,
        @Nullable final StopWatch stopWatch,
        final String prefixMessage,
        final int shardId,
        final String segmentName,
        final String field
    ) {

        if (stopWatch != null && log.isDebugEnabled()) {
            try {
                stopWatch.stop();
                final String logMessage = prefixMessage + " shard: [{}], segment: [{}], field: [{}], time in nanos:[{}] ";
                log.debug(logMessage, shardId, segmentName, field, stopWatch.totalTime().nanos());
            } catch (IllegalStateException e) {
                log.error(
                    "StopWatch was already stopped for operation: {} on shard: [{}], segment: [{}], field: [{}]",
                    prefixMessage,
                    shardId,
                    segmentName,
                    field
                );
            }
        }
    }
}
