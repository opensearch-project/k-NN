/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import lombok.extern.log4j.Log4j2;

import java.lang.reflect.Field;

@Log4j2
public class KNNMergeHelper {

    private static final Field MERGE_FIELD;
    static {
        try {
            MERGE_FIELD = ConcurrentMergeScheduler.MergeThread.class.getDeclaredField("merge");
            MERGE_FIELD.setAccessible(true);
        } catch (NoSuchFieldException e) {
            throw new RuntimeException("Failed to initialize", e);
        }
    }

    private KNNMergeHelper() {}

    public static boolean isMergeAborted() {
        Thread mergeThread = Thread.currentThread();
        if (mergeThread instanceof ConcurrentMergeScheduler.MergeThread) {
            try {
                MergePolicy.OneMerge merge = (MergePolicy.OneMerge) MERGE_FIELD.get(mergeThread);
                return merge.isAborted();
            } catch (RuntimeException e) {
                log.error("isMergeAborted get exception", e);
                return false;
            } catch (IllegalAccessException e) {
                log.error("IllegalAccessException for reflection merge field", e);
                return false;
            }
        }
        return false;
    }
}
