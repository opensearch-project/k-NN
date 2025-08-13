/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.lucene.store.Directory;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

public class KNNIOUtilsTests extends KNNTestCase {

    public void testCloseWhileSuppressingExceptions_singleThread_successfulClose() throws Exception {
        AtomicInteger counter = new AtomicInteger();
        Closeable c1 = () -> counter.incrementAndGet();
        Closeable c2 = () -> counter.incrementAndGet();

        Throwable base = new RuntimeException("Base exception");
        KNNIOUtils.closeWhileSuppressingExceptions(base, c1, null, c2); // Includes null

        assertEquals(2, counter.get());
    }

    public void testCloseWhileSuppressingExceptions_singleThread_withSuppressed() {
        Closeable badClose = () -> { throw new IOException("Close failed"); };
        Throwable base = new RuntimeException("Base");

        base.setStackTrace(new StackTraceElement[0]); // Clean up stack trace
        KNNIOUtils.closeWhileSuppressingExceptions(base, badClose);

        assertEquals(1, base.getSuppressed().length);
        assertEquals("Close failed", base.getSuppressed()[0].getMessage());
    }

    public void testDeleteFilesSuppressingExceptions_singleThread_successfulDelete() throws IOException {
        Directory dir = mock(Directory.class);
        List<String> files = List.of("file1", "file2");

        Throwable base = new RuntimeException("Base");
        KNNIOUtils.deleteFilesSuppressingExceptions(base, dir, files);

        for (String f : files) {
            verify(dir).deleteFile(f);
        }
    }

    public void testDeleteFilesSuppressingExceptions_singleThread_withSuppressed() throws IOException {
        Directory dir = mock(Directory.class);
        doThrow(new IOException("Delete failed")).when(dir).deleteFile(anyString());

        Throwable base = new RuntimeException("Base");
        KNNIOUtils.deleteFilesSuppressingExceptions(base, dir, "fileA");

        assertEquals(1, base.getSuppressed().length);
        assertEquals("Delete failed", base.getSuppressed()[0].getMessage());
    }

    public void testCloseWhileSuppressingExceptions_concurrentExecution() throws Exception {
        int threadCount = 10;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        AtomicInteger counter = new AtomicInteger();
        List<Future<?>> futures = new ArrayList<>();

        class TestCloseable implements Closeable {
            public void close() {
                counter.incrementAndGet();
            }
        }

        for (int i = 0; i < threadCount; i++) {
            futures.add(executor.submit(() -> {
                Throwable ex = new RuntimeException("ThreadEx");
                KNNIOUtils.closeWhileSuppressingExceptions(ex, new TestCloseable(), new TestCloseable());
            }));
        }

        for (Future<?> f : futures) {
            f.get(); // Waits for all threads to finish
        }

        executor.shutdownNow();
        assertEquals(threadCount * 2, counter.get());
    }

    public void testDeleteFilesSuppressingExceptions_concurrentExecution() throws Exception {
        int threadCount = 5;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        Set<String> deletedFiles = ConcurrentHashMap.newKeySet();
        Directory dir = mock(Directory.class);

        doAnswer(invocation -> {
            String file = invocation.getArgument(0);
            deletedFiles.add(file);
            return null;
        }).when(dir).deleteFile(anyString());

        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < threadCount; i++) {
            int threadIndex = i;
            futures.add(executor.submit(() -> {
                Throwable ex = new RuntimeException("Ex");
                KNNIOUtils.deleteFilesSuppressingExceptions(ex, dir, "f1-" + threadIndex, "f2-" + threadIndex);
            }));
        }

        for (Future<?> f : futures) {
            f.get();
        }

        executor.shutdownNow();
        assertEquals(threadCount * 2, deletedFiles.size());
    }

    public void testDeleteFilesSuppressingExceptions_varargsOverload() throws IOException {
        Directory dir = mock(Directory.class);
        Throwable base = new RuntimeException("Base");

        KNNIOUtils.deleteFilesSuppressingExceptions(base, dir, "file1", "file2", "file3");

        verify(dir).deleteFile("file1");
        verify(dir).deleteFile("file2");
        verify(dir).deleteFile("file3");
    }

    public void testCloseWhileSuppressingExceptions_concurrent_withNulls() throws Exception {
        int threadCount = 5;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        AtomicInteger counter = new AtomicInteger();
        List<Future<?>> futures = new ArrayList<>();

        class TestCloseable implements Closeable {
            @Override
            public void close() {
                counter.incrementAndGet();
            }
        }

        for (int i = 0; i < threadCount; i++) {
            futures.add(executor.submit(() -> {
                Throwable ex = new RuntimeException("ThreadEx");
                KNNIOUtils.closeWhileSuppressingExceptions(ex, new TestCloseable(), null, new TestCloseable(), null);
            }));
        }

        for (Future<?> f : futures) {
            f.get();
        }

        executor.shutdownNow();
        assertEquals(threadCount * 2, counter.get()); // Each thread has 2 non-null closeables
    }

    public void testCloseWhileSuppressingExceptions_concurrent_withFailures() throws Exception {
        int threadCount = 3;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        List<Future<Throwable>> futures = new ArrayList<>();

        class FailingCloseable implements Closeable {
            private final String name;

            FailingCloseable(String name) {
                this.name = name;
            }

            @Override
            public void close() throws IOException {
                throw new IOException("Failure in " + name);
            }
        }

        for (int i = 0; i < threadCount; i++) {
            int idx = i;
            futures.add(executor.submit(() -> {
                Throwable ex = new RuntimeException("Base " + idx);
                KNNIOUtils.closeWhileSuppressingExceptions(ex, new FailingCloseable("A" + idx), null, new FailingCloseable("B" + idx));
                return ex;
            }));
        }

        for (Future<Throwable> f : futures) {
            Throwable base = f.get();
            assertEquals(2, base.getSuppressed().length); // Each thread had 2 failing closeables
        }

        executor.shutdownNow();
    }

    public void testCloseWhileSuppressingExceptions_concurrent_mixed() throws Exception {
        int threadCount = 4;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        AtomicInteger counter = new AtomicInteger();
        List<Future<Throwable>> futures = new ArrayList<>();

        class SafeCloseable implements Closeable {
            @Override
            public void close() {
                counter.incrementAndGet();
            }
        }

        class FailingCloseable implements Closeable {
            @Override
            public void close() throws IOException {
                throw new IOException("Deliberate failure");
            }
        }

        for (int i = 0; i < threadCount; i++) {
            futures.add(executor.submit(() -> {
                Throwable base = new RuntimeException("Base");
                KNNIOUtils.closeWhileSuppressingExceptions(
                    base,
                    new SafeCloseable(),
                    new FailingCloseable(),
                    null,
                    new SafeCloseable(),
                    new FailingCloseable()
                );
                return base;
            }));
        }

        for (Future<Throwable> f : futures) {
            Throwable ex = f.get();
            assertEquals(2, ex.getSuppressed().length); // 2 failures per thread
        }

        assertEquals(threadCount * 2, counter.get()); // 2 successful closes per thread
        executor.shutdownNow();
    }
}
