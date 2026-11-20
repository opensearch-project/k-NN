/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */
package org.opensearch.knn.common;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.ActionType;
import org.opensearch.common.CheckedRunnable;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.identity.PluginSubject;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.client.Client;

import java.security.Principal;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Tests that PluginClient runs actions as its subject and notifies the listener in the caller's context. */
public class PluginClientTests extends OpenSearchTestCase {

    private static final String CALLER_TRANSIENT = "caller.marker";
    private static final String SUBJECT_TRANSIENT = "subject.marker";
    private static final String RESPONDER_TRANSIENT = "responder.marker";

    private static final ActionType<DummyResponse> DUMMY_ACTION = new ActionType<>(
        "cluster:admin/plugin_client_test",
        in -> new DummyResponse()
    );

    private ThreadContext threadContext;
    private Client delegate;
    private PluginClient pluginClient;
    private ExecutorService responder;

    private ActionListener<DummyResponse> capturedListener;
    private String subjectSeenByDelegate;
    private String callerSeenByDelegate;
    private Consumer<ActionListener<DummyResponse>> delegateBehaviour;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        threadContext = new ThreadContext(Settings.EMPTY);
        responder = Executors.newSingleThreadExecutor();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.getThreadContext()).thenReturn(threadContext);

        delegate = mock(Client.class);
        when(delegate.settings()).thenReturn(Settings.EMPTY);
        when(delegate.threadPool()).thenReturn(threadPool);
        // Non-null so that a client forwarding the call would return something instead of throwing.
        when(delegate.getRemoteClusterClient(anyString())).thenReturn(mock(Client.class));

        // Records the context the action ran in and keeps the listener, so each test decides when and
        // how the action completes. By default it does not complete, as for an asynchronous action.
        delegateBehaviour = listener -> {};
        doAnswer(invocation -> {
            subjectSeenByDelegate = threadContext.getTransient(SUBJECT_TRANSIENT);
            callerSeenByDelegate = threadContext.getTransient(CALLER_TRANSIENT);
            capturedListener = invocation.getArgument(2);
            delegateBehaviour.accept(capturedListener);
            return null;
        }).when(delegate).execute(any(ActionType.class), any(ActionRequest.class), any(ActionListener.class));

        pluginClient = new PluginClient(delegate);
    }

    @Override
    public void tearDown() throws Exception {
        responder.shutdownNow();
        assertTrue(responder.awaitTermination(10, TimeUnit.SECONDS));
        super.tearDown();
    }

    public void testRunsTheActionInsideTheSubject() {
        threadContext.putTransient(CALLER_TRANSIENT, "set-by-caller");
        assignSubject();

        pluginClient.execute(DUMMY_ACTION, new DummyRequest(), ActionListener.wrap(r -> {}, e -> {}));

        assertEquals("the action must run inside runAs", "set-by-subject", subjectSeenByDelegate);
        assertNull("the caller's context must not reach the action", callerSeenByDelegate);
        assertEquals("the calling thread must get its context back", "set-by-caller", threadContext.getTransient(CALLER_TRANSIENT));
        assertNull("the subject's context must not stay on the calling thread", threadContext.getTransient(SUBJECT_TRANSIENT));
    }

    public void testListenerSeesTheCallerContextAndNotTheSubjects() throws Exception {
        assertListenerSeesCallerContext(listener -> listener.onResponse(new DummyResponse()));
    }

    public void testListenerSeesTheCallerContextOnFailure() throws Exception {
        assertListenerSeesCallerContext(listener -> listener.onFailure(new RuntimeException("action failed")));
    }

    public void testRestoresTheRespondingThreadsContextAfterTheListenerReturns() throws Exception {
        threadContext.putTransient(CALLER_TRANSIENT, "set-by-caller");
        assignSubject();
        pluginClient.execute(DUMMY_ACTION, new DummyRequest(), ActionListener.wrap(r -> {}, e -> {}));

        // Restoring the caller's context and leaving it there would leave it on the responding thread
        // for whatever that pooled thread runs next.
        AtomicReference<String> responderAfter = new AtomicReference<>();
        AtomicReference<String> callerLeftBehind = new AtomicReference<>();
        completeOnAnotherThread(listener -> {
            listener.onResponse(new DummyResponse());
            responderAfter.set(threadContext.getTransient(RESPONDER_TRANSIENT));
            callerLeftBehind.set(threadContext.getTransient(CALLER_TRANSIENT));
        });

        assertEquals("the responding thread's own context must come back", "set-by-responder", responderAfter.get());
        assertNull("the caller's context must not be left on the responding thread", callerLeftBehind.get());
    }

    public void testListenerSeesResponseHeadersTheActionAdded() throws Exception {
        assignSubject();
        AtomicReference<List<String>> warnings = new AtomicReference<>();
        pluginClient.execute(
            DUMMY_ACTION,
            new DummyRequest(),
            ActionListener.wrap(r -> warnings.set(threadContext.getResponseHeaders().get("Warning")), e -> {})
        );

        completeOnAnotherThread(listener -> {
            threadContext.addResponseHeader("Warning", "deprecated");
            listener.onResponse(new DummyResponse());
        });

        assertEquals(List.of("deprecated"), warnings.get());
    }

    public void testThrowsADelegateFailureToTheCallerAsTheWrappedClientWould() {
        assignSubject();
        RuntimeException boom = new RuntimeException("failed to find action");
        delegateBehaviour = listener -> { throw boom; };

        List<Exception> failures = new ArrayList<>();
        RuntimeException thrown = expectThrows(
            RuntimeException.class,
            () -> pluginClient.execute(DUMMY_ACTION, new DummyRequest(), ActionListener.wrap(r -> {}, failures::add))
        );

        assertSame(boom, thrown);
        assertEquals("a failure thrown before the listener exists must not also reach it", List.of(), failures);
    }

    public void testDoesNotNotifyAgainWhenTheListenerItselfThrows() {
        assignSubject();
        delegateBehaviour = listener -> listener.onResponse(new DummyResponse());

        RuntimeException listenerBug = new RuntimeException("listener bug");
        AtomicInteger calls = new AtomicInteger();
        ActionListener<DummyResponse> throwing = new ActionListener<>() {
            @Override
            public void onResponse(DummyResponse response) {
                calls.incrementAndGet();
                throw listenerBug;
            }

            @Override
            public void onFailure(Exception e) {
                calls.incrementAndGet();
            }
        };

        RuntimeException surfaced = expectThrows(
            RuntimeException.class,
            () -> pluginClient.execute(DUMMY_ACTION, new DummyRequest(), throwing)
        );
        assertSame(listenerBug, surfaced);
        assertEquals("the listener must be told exactly once", 1, calls.get());
    }

    public void testRejectsUseBeforeASubjectIsAssigned() {
        IllegalStateException e = expectThrows(
            IllegalStateException.class,
            () -> pluginClient.execute(DUMMY_ACTION, new DummyRequest(), ActionListener.wrap(r -> {}, ex -> {}))
        );
        assertTrue(e.getMessage().contains("subject"));
    }

    public void testAcceptsTheSameSubjectAgainButRejectsADifferentOne() {
        RecordingSubject first = assignSubject();
        pluginClient.setSubject(first);

        expectThrows(IllegalStateException.class, () -> pluginClient.setSubject(new RecordingSubject(threadContext)));
    }

    public void testRejectsANullSubject() {
        expectThrows(NullPointerException.class, () -> pluginClient.setSubject(null));
    }

    public void testRefusesARemoteClusterClient() {
        expectThrows(UnsupportedOperationException.class, () -> pluginClient.getRemoteClusterClient("remote"));
    }

    private void assertListenerSeesCallerContext(Consumer<ActionListener<DummyResponse>> completion) throws Exception {
        threadContext.putTransient(CALLER_TRANSIENT, "set-by-caller");
        assignSubject();

        AtomicReference<String> callerSeen = new AtomicReference<>("listener did not run");
        AtomicReference<String> subjectSeen = new AtomicReference<>("listener did not run");
        Consumer<Object> record = ignored -> {
            callerSeen.set(threadContext.getTransient(CALLER_TRANSIENT));
            subjectSeen.set(threadContext.getTransient(SUBJECT_TRANSIENT));
        };
        pluginClient.execute(DUMMY_ACTION, new DummyRequest(), ActionListener.wrap(record::accept, record::accept));

        // A response arrives on a pooled thread that never carried the caller's context, so the wrapper
        // around the listener is the only thing that can put it back.
        completeOnAnotherThread(completion);

        assertEquals("set-by-caller", callerSeen.get());
        assertNull("the subject's context must not reach the listener", subjectSeen.get());
    }

    private RecordingSubject assignSubject() {
        RecordingSubject subject = new RecordingSubject(threadContext);
        pluginClient.setSubject(subject);
        return subject;
    }

    /** Completes the captured action on a thread that never carried the caller's context. */
    private void completeOnAnotherThread(Consumer<ActionListener<DummyResponse>> completion) throws Exception {
        responder.submit(() -> {
            threadContext.putTransient(RESPONDER_TRANSIENT, "set-by-responder");
            completion.accept(capturedListener);
        }).get(10, TimeUnit.SECONDS);
    }

    /**
     * Stands in for the plugin subject. Stashes and restores around its body as NoopPluginSubject and
     * SecurePluginSubject do, and marks the context so a test can tell the subject's apart from the
     * caller's.
     */
    private static class RecordingSubject implements PluginSubject {

        private final ThreadContext threadContext;

        RecordingSubject(ThreadContext threadContext) {
            this.threadContext = threadContext;
        }

        @Override
        public Principal getPrincipal() {
            return () -> "plugin:test";
        }

        @Override
        public <E extends Exception> void runAs(CheckedRunnable<E> runnable) throws E {
            try (ThreadContext.StoredContext ignored = threadContext.stashContext()) {
                threadContext.putTransient(SUBJECT_TRANSIENT, "set-by-subject");
                runnable.run();
            }
        }
    }

    private static class DummyRequest extends ActionRequest {
        @Override
        public ActionRequestValidationException validate() {
            return null;
        }
    }

    private static class DummyResponse extends ActionResponse {
        @Override
        public void writeTo(StreamOutput out) {}
    }
}
