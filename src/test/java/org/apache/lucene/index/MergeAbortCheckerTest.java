/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import org.junit.Test;
import org.mockito.MockedStatic;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link MergeAbortChecker}.
 */
public class MergeAbortCheckerTest {

    @Test
    public void testIsMergeAborted_NotMergeThread() {
        // Test with regular thread (not a merge thread)
        assertFalse(MergeAbortChecker.isMergeAborted());
    }

    @Test
    public void testIsMergeAborted_MergeThreadNotAborted() throws Exception {
        // Create mock merge thread and merge
        ConcurrentMergeScheduler.MergeThread mockMergeThread = mock(ConcurrentMergeScheduler.MergeThread.class);
        MergePolicy.OneMerge mockMerge = mock(MergePolicy.OneMerge.class);

        when(mockMerge.isAborted()).thenReturn(false);

        // Mock the reflection field access
        Field mergeField = ConcurrentMergeScheduler.MergeThread.class.getDeclaredField("merge");
        mergeField.setAccessible(true);
        mergeField.set(mockMergeThread, mockMerge);

        try (MockedStatic<Thread> threadMock = mockStatic(Thread.class)) {
            threadMock.when(Thread::currentThread).thenReturn(mockMergeThread);

            assertFalse(MergeAbortChecker.isMergeAborted());
        }
    }

    @Test
    public void testIsMergeAborted_MergeThreadAborted() throws Exception {
        // Create mock merge thread and merge
        ConcurrentMergeScheduler.MergeThread mockMergeThread = mock(ConcurrentMergeScheduler.MergeThread.class);
        MergePolicy.OneMerge mockMerge = mock(MergePolicy.OneMerge.class);

        when(mockMerge.isAborted()).thenReturn(true);

        // Mock the reflection field access
        Field mergeField = ConcurrentMergeScheduler.MergeThread.class.getDeclaredField("merge");
        mergeField.setAccessible(true);
        mergeField.set(mockMergeThread, mockMerge);

        try (MockedStatic<Thread> threadMock = mockStatic(Thread.class)) {
            threadMock.when(Thread::currentThread).thenReturn(mockMergeThread);

            assertTrue(MergeAbortChecker.isMergeAborted());
        }
    }

    @Test
    public void testConstructorIsPrivate() throws Exception {
        // Verify constructor is private
        Constructor<MergeAbortChecker> constructor = MergeAbortChecker.class.getDeclaredConstructor();
        assertTrue(Modifier.isPrivate(constructor.getModifiers()));

        // Verify it can be invoked via reflection
        constructor.setAccessible(true);
        constructor.newInstance();
    }

    @Test
    public void testIsMergeAborted_RuntimeException() throws Exception {
        // Create mock merge thread and merge that throws RuntimeException
        ConcurrentMergeScheduler.MergeThread mockMergeThread = mock(ConcurrentMergeScheduler.MergeThread.class);
        MergePolicy.OneMerge mockMerge = mock(MergePolicy.OneMerge.class);

        when(mockMerge.isAborted()).thenThrow(new RuntimeException("Test exception"));

        Field mergeField = ConcurrentMergeScheduler.MergeThread.class.getDeclaredField("merge");
        mergeField.setAccessible(true);
        mergeField.set(mockMergeThread, mockMerge);

        try (MockedStatic<Thread> threadMock = mockStatic(Thread.class)) {
            threadMock.when(Thread::currentThread).thenReturn(mockMergeThread);

            // Should return false on exception
            assertFalse(MergeAbortChecker.isMergeAborted());
        }
    }

    @Test
    public void testIsMergeAborted_NullMerge() throws Exception {
        // Create mock merge thread with null merge field
        ConcurrentMergeScheduler.MergeThread mockMergeThread = mock(ConcurrentMergeScheduler.MergeThread.class);

        Field mergeField = ConcurrentMergeScheduler.MergeThread.class.getDeclaredField("merge");
        mergeField.setAccessible(true);
        mergeField.set(mockMergeThread, null);

        try (MockedStatic<Thread> threadMock = mockStatic(Thread.class)) {
            threadMock.when(Thread::currentThread).thenReturn(mockMergeThread);

            // Should return false when merge is null
            assertFalse(MergeAbortChecker.isMergeAborted());
        }
    }
}
