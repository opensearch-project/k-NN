/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;
import org.opensearch.core.common.io.stream.Writeable;

/**
 * Action associated with ClearCache
 */
public class ClearCacheAction extends ActionType<ClearCacheResponse> {

    public static final ClearCacheAction INSTANCE = new ClearCacheAction();
    public static final String NAME = "cluster:admin/clear_cache_action";

    private ClearCacheAction() {
        super(NAME, ClearCacheResponse::new);
    }

    @Override
    public Writeable.Reader<ClearCacheResponse> getResponseReader() {
        return ClearCacheResponse::new;
    }
}
