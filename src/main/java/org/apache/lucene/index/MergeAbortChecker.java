/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import lombok.extern.log4j.Log4j2;

import java.lang.reflect.Field;

/**
 * Utility class for checking if a Lucene segment merge operation has been aborted.
 * Uses reflection to access the merge state from {@link ConcurrentMergeScheduler.MergeThread}.
 *
 * <p>This is used in faiss interrupt callback mechanism to be able to abort faiss index creation
 * when merge process is aborted by Lucene's {@link IndexWriter}.
 *
 * @since 1.0
 */
@Log4j2
public final class MergeAbortChecker {

    /** The reflected field for accessing merge state from MergeThread */
    private static final Field MERGE_FIELD;

    /** The name of the merge field in ConcurrentMergeScheduler.MergeThread */
    private static final String MERGE_FIELD_NAME = "merge";

    static {
        try {
            MERGE_FIELD = ConcurrentMergeScheduler.MergeThread.class.getDeclaredField(MERGE_FIELD_NAME);
            MERGE_FIELD.setAccessible(true);
        } catch (NoSuchFieldException e) {
            throw new RuntimeException("Failed to initialize MergeAbortChecker: merge field not found", e);
        }
    }

    /**
     * Private constructor to prevent instantiation of utility class.
     */
    private MergeAbortChecker() {}

    /**
     * Checks if the current thread is a merge thread and if its merge operation has been aborted.
     *
     * <p>This method uses reflection to access the private merge field from
     * {@link ConcurrentMergeScheduler.MergeThread} and checks if the associated
     * {@link MergePolicy.OneMerge} has been aborted.
     *
     * @return {@code true} if the current thread is a {@link ConcurrentMergeScheduler.MergeThread}
     *         and its merge is aborted, {@code false} otherwise (including when not running
     *         in a merge thread context or when reflection access fails)
     *
     * @see ConcurrentMergeScheduler.MergeThread
     * @see MergePolicy.OneMerge#isAborted()
     */
    public static boolean isMergeAborted() {
        Thread mergeThread = Thread.currentThread();
        if (mergeThread instanceof ConcurrentMergeScheduler.MergeThread) {
            try {
                MergePolicy.OneMerge merge = (MergePolicy.OneMerge) MERGE_FIELD.get(mergeThread);
                boolean aborted = merge.isAborted();
                log.debug("Current Thread {} is aborted: {}", mergeThread.getName(), aborted);
                return aborted;
            } catch (RuntimeException e) {
                log.error("Runtime exception while checking merge abort status", e);
                return false;
            } catch (IllegalAccessException e) {
                log.error("IllegalAccessException for reflection merge field access", e);
                return false;
            }
        }
        return false;
    }
}
