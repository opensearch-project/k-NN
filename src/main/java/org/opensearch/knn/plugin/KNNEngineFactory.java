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

package org.opensearch.knn.plugin;

import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.index.engine.InternalEngine;

/**
 * EngineFactory to inject the KNNCodecService to help segments write using the KNNCodec.
 */
class KNNEngineFactory implements EngineFactory {

    private static KNNCodecService codecService = new KNNCodecService();

    @Override
    public Engine newReadWriteEngine(EngineConfig config) {
        codecService.setPostingsFormat(config.getCodec().postingsFormat());
        EngineConfig engineConfig = new EngineConfig(config.getShardId(),
                config.getThreadPool(), config.getIndexSettings(), config.getWarmer(), config.getStore(),
                config.getMergePolicy(), config.getAnalyzer(), config.getSimilarity(), codecService,
                config.getEventListener(), config.getQueryCache(), config.getQueryCachingPolicy(),
                config.getTranslogConfig(), config.getFlushMergesAfter(), config.getExternalRefreshListener(),
                config.getInternalRefreshListener(), config.getIndexSort(), config.getCircuitBreakerService(),
                config.getGlobalCheckpointSupplier(), config.retentionLeasesSupplier(), config.getPrimaryTermSupplier(),
                config.getTombstoneDocSupplier());
        return new InternalEngine(engineConfig);
    }
}
