/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;
import org.opensearch.core.common.io.stream.Writeable;

/**
 * Action used to remove a model from the cache on some or all nodes
 */
public class RemoveModelFromCacheAction extends ActionType<RemoveModelFromCacheResponse> {

    public static final String NAME = "cluster:admin/knn_remove_model_from_cache_action";
    public static final RemoveModelFromCacheAction INSTANCE = new RemoveModelFromCacheAction(NAME, RemoveModelFromCacheResponse::new);

    /**
     * Constructor
     *
     * @param name name of action
     * @param responseReader reader for the RemoveModelFromCacheResponse response
     */
    public RemoveModelFromCacheAction(String name, Writeable.Reader<RemoveModelFromCacheResponse> responseReader) {
        super(name, responseReader);
    }
}
