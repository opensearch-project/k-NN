/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */
package org.opensearch.knn.common;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionType;
import org.opensearch.action.support.ContextPreservingActionListener;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.identity.PluginSubject;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.FilterClient;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Runs transport actions as the k-NN plugin's system subject rather than as the authenticated user, so
 * the plugin can reach the model system index it registers.
 * <p>
 * Core assigns the subject through {@code IdentityAwarePlugin.assignSubject} after
 * {@code createComponents}, with or without the security plugin installed. Apart from the identity it
 * runs as, this client behaves like the client it wraps, including which failures are thrown and which
 * reach the listener.
 */
public final class PluginClient extends FilterClient {

    private static final Logger log = LogManager.getLogger(PluginClient.class);

    // Set once, from assignSubject, on a different thread from the actions that read it.
    private final AtomicReference<PluginSubject> subject = new AtomicReference<>();

    /**
     * Wraps a client so its actions run as the plugin subject, which is assigned later.
     *
     * @param delegate the client whose actions this wraps
     */
    public PluginClient(Client delegate) {
        super(delegate);
    }

    /**
     * Assigns the subject every action runs as. It can be assigned once; assigning the same subject
     * again is a no-op.
     *
     * @param subject the plugin's system subject, normally the one passed to {@code assignSubject}
     * @throws IllegalStateException if a different subject is already assigned
     */
    public void setSubject(PluginSubject subject) {
        Objects.requireNonNull(subject, "subject");
        PluginSubject existing = this.subject.compareAndExchange(null, subject);
        if (existing != null && existing != subject) {
            throw new IllegalStateException("PluginClient already has a subject assigned.");
        }
    }

    /**
     * Not supported, because a plugin subject has no meaning on another cluster.
     *
     * @param clusterAlias the remote cluster that was requested
     * @return never returns
     * @throws UnsupportedOperationException always
     */
    @Override
    public Client getRemoteClusterClient(String clusterAlias) {
        throw new UnsupportedOperationException("PluginClient cannot run actions on a remote cluster.");
    }

    /**
     * Runs the action as the assigned subject.
     *
     * @param action the action to run as the subject
     * @param request the request to run it with
     * @param listener notified with the caller's thread context in place rather than the subject's
     * @throws IllegalStateException if no subject has been assigned yet, which is a wiring error
     */
    @Override
    protected <Request extends ActionRequest, Response extends ActionResponse> void doExecute(
        ActionType<Response> action,
        Request request,
        ActionListener<Response> listener
    ) {
        PluginSubject currentSubject = subject.get();
        if (currentSubject == null) {
            throw new IllegalStateException("PluginClient is not initialized with a subject.");
        }

        // Captured before runAs stashes the context; core restores it around the callback, then puts
        // the responding thread's own context back.
        ActionListener<Response> callerContext = ContextPreservingActionListener.wrapPreservingContext(
            listener,
            threadPool().getThreadContext()
        );
        currentSubject.runAs(() -> {
            log.debug("Running transport action as subject: {}", currentSubject.getPrincipal().getName());
            super.doExecute(action, request, callerContext);
        });
    }
}
