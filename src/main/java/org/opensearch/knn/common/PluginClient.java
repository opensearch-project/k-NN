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
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.identity.Subject;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.FilterClient;

/**
 * Executes transport actions as this plugin's assigned system subject rather than as the
 * authenticated user, which is how the plugin reaches the system indices it owns.
 */
public class PluginClient extends FilterClient {

    private static final Logger log = LogManager.getLogger(PluginClient.class);

    // Assigned from IdentityAwarePlugin.assignSubject, which runs on a different thread than the
    // transport actions that read it.
    private volatile Subject subject;

    public PluginClient(Client delegate) {
        super(delegate);
    }

    public void setSubject(Subject subject) {
        this.subject = subject;
    }

    @Override
    protected <Request extends ActionRequest, Response extends ActionResponse> void doExecute(
        ActionType<Response> action,
        Request request,
        ActionListener<Response> listener
    ) {
        Subject currentSubject = this.subject;
        if (currentSubject == null) {
            throw new IllegalStateException("PluginClient is not initialized with a subject.");
        }

        // Saves the caller's context so it can be restored once the action completes. runAs performs
        // the switch itself, so stashing here would clear the context before it gets the chance.
        // Held in a local rather than a try-with-resources because the action is asynchronous: a try
        // block would restore on exit, long before the listener fires.
        ThreadContext.StoredContext storedContext = threadPool().getThreadContext().newStoredContext(false);

        try {
            currentSubject.runAs(() -> {
                log.debug("Running transport action as subject: {}", currentSubject.getPrincipal().getName());
                super.doExecute(action, request, ActionListener.runBefore(listener, storedContext::restore));
            });
        } catch (Exception e) {
            // Reported through the listener rather than thrown, so an async caller is not left waiting.
            storedContext.close();
            listener.onFailure(e);
        }
    }
}
