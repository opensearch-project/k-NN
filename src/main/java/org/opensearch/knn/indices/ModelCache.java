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

package org.opensearch.knn.indices;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.RemovalCause;
import com.google.common.cache.RemovalNotification;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.cluster.service.ClusterService;

import java.time.Instant;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.MODEL_CACHE_EXPIRE_AFTER_ACCESS_TIME_MINUTES;
import static org.opensearch.knn.index.KNNSettings.MODEL_CACHE_SIZE_LIMIT_SETTING;

public final class ModelCache {

    private static Logger logger = LogManager.getLogger(ModelCache.class);

    private static ModelCache instance;
    private static ModelDao modelDao;
    private static ClusterService clusterService;

    private Cache<String, Model> cache;
    private long cacheSizeInKB;
    private Instant evictedDueToSizeAt;

    /**
     * Get instance of cache
     *
     * @return singleton instance of cache
     */
    public static synchronized ModelCache getInstance() {
        if (instance == null) {
            instance = new ModelCache();
        }
        return instance;
    }

    /**
     * Initialize the cache
     *
     * @param modelDao modelDao used to read persistence layer for models
     * @param clusterService used to update settings
     */
    public static void initialize(ModelDao modelDao, ClusterService clusterService) {
        ModelCache.modelDao = modelDao;
        ModelCache.clusterService = clusterService;
    }

    /**
     * Evict all entries and rebuild the graph
     */
    public synchronized void rebuild() {
        cache.invalidateAll();
        initCache();
    }

    protected ModelCache() {
        cacheSizeInKB = MODEL_CACHE_SIZE_LIMIT_SETTING.get(clusterService.getSettings()).getKb();
        clusterService.getClusterSettings().addSettingsUpdateConsumer(MODEL_CACHE_SIZE_LIMIT_SETTING, it -> {
            cacheSizeInKB = it.getKb();
            rebuild();
        });
        initCache();
    }

    private void initCache() {
        CacheBuilder<String, Model> cacheBuilder = CacheBuilder.newBuilder()
            .recordStats()
            .concurrencyLevel(1)
            .removalListener(this::onRemoval)
            .maximumWeight(cacheSizeInKB)
            .expireAfterAccess(MODEL_CACHE_EXPIRE_AFTER_ACCESS_TIME_MINUTES, TimeUnit.MINUTES)
            .weigher((k, v) -> Math.toIntExact(getModelLengthInKB(v)));

        cache = cacheBuilder.build();
    }

    private void onRemoval(RemovalNotification<String, Model> removalNotification) {
        if (RemovalCause.SIZE == removalNotification.getCause()) {
            updateEvictedDueToSizeAt();
        }

        logger.info("[KNN] Model Cache evicted. Key {}, Reason: {}", removalNotification.getKey(), removalNotification.getCause());
    }

    public Instant getEvictedDueToSizeAt() {
        return evictedDueToSizeAt;
    }

    private void updateEvictedDueToSizeAt() {
        this.evictedDueToSizeAt = Instant.now();
    }

    /**
     * Get the model from modelId
     *
     * @param modelId model identifier
     * @return Model Entry representing model
     */
    public Model get(String modelId) {
        try {
            return cache.get(modelId, () -> modelDao.get(modelId));
        } catch (ExecutionException ee) {
            throw new IllegalStateException("Unable to retrieve model binary for \"" + modelId + "\": " + ee);
        }
    }

    /**
     * Get total weight of cache
     *
     * @return total weight
     */
    public long getTotalWeightInKB() {
        return cache.asMap().values().stream().map(this::getModelLengthInKB).reduce(0L, Long::sum);
    }

    /**
     * Remove modelId from cache
     *
     * @param modelId to be removed
     */
    public void remove(String modelId) {
        cache.invalidate(modelId);
    }

    /**
     * Check if modelId is in the cache
     *
     * @param modelId model id to be checked
     * @return true if model id is in the cache; false otherwise
     */
    public boolean contains(String modelId) {
        return cache.asMap().containsKey(modelId);
    }

    /**
     * Remove all elements from the cache
     */
    public void removeAll() {
        cache.invalidateAll();
    }

    private Long getModelLengthInKB(Model model) {
        return (model.getLength() / BYTES_PER_KILOBYTES) + 1L;
    }
}
