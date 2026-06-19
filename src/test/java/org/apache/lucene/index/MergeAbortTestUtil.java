/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Test-only utility that executes a task on a real {@link ConcurrentMergeScheduler.MergeThread}
 * whose {@link MergePolicy.OneMerge} has been aborted.
 *
 * <p>This lets tests exercise the genuine {@link MergeAbortChecker#isMergeAborted()} reflection
 * path (the same path used in production during a merge) instead of statically mocking
 * {@code MergeAbortChecker}. Static mocking is unreliable here because the merge-abort check is
 * invoked by native faiss code through JNI ({@code CallStaticBooleanMethod}), and inline static
 * mocks are not consistently honored for native-originated calls across all JDK/runtime
 * combinations. Driving the real merge-thread state is portable and faithfully reproduces what the
 * managed service experiences when a merge is aborted.
 *
 * <p>This class lives in the {@code org.apache.lucene.index} package so it can access the
 * package-private {@code ConcurrentMergeScheduler.MergeThread} and its {@code merge} field.
 */
public final class MergeAbortTestUtil {

    private MergeAbortTestUtil() {}

    /** A task that may throw any {@link Throwable}. */
    @FunctionalInterface
    public interface ThrowingRunnable {
        void run() throws Throwable;
    }

    /**
     * Runs {@code task} on a {@link ConcurrentMergeScheduler.MergeThread} backed by an aborted
     * merge, then returns any {@link Throwable} the task threw (or {@code null} if it completed
     * normally). While the task runs, {@link MergeAbortChecker#isMergeAborted()} returns
     * {@code true} for the current thread.
     */
    public static Throwable runOnAbortedMergeThread(final ThrowingRunnable task) throws InterruptedException {
        final ConcurrentMergeScheduler scheduler = new ConcurrentMergeScheduler();

        // An empty OneMerge is sufficient: we only need a non-null merge whose aborted flag is set.
        final MergePolicy.OneMerge merge = new MergePolicy.OneMerge();
        merge.setAborted();

        final MergeScheduler.MergeSource mergeSource = new MergeScheduler.MergeSource() {
            @Override
            public MergePolicy.OneMerge getNextMerge() {
                return null;
            }

            @Override
            public void onMergeFinished(MergePolicy.OneMerge oneMerge) {}

            @Override
            public boolean hasPendingMerges() {
                return false;
            }

            @Override
            public void merge(MergePolicy.OneMerge oneMerge) {}
        };

        final AtomicReference<Throwable> thrown = new AtomicReference<>();
        final ConcurrentMergeScheduler.MergeThread mergeThread = scheduler.new MergeThread(mergeSource, merge) {
            @Override
            public void run() {
                try {
                    task.run();
                } catch (Throwable t) {
                    thrown.set(t);
                }
            }
        };

        mergeThread.start();
        mergeThread.join();
        return thrown.get();
    }
}
