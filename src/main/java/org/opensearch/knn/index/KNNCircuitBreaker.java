/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.OpenSearchParseException;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.monitor.jvm.JvmInfo;
import org.opensearch.monitor.os.OsProbe;
import org.opensearch.threadpool.ThreadPool;

import java.util.List;
import java.util.concurrent.ExecutionException;

import static org.opensearch.common.settings.Setting.Property.Dynamic;
import static org.opensearch.common.settings.Setting.Property.NodeScope;
import static org.opensearch.core.common.unit.ByteSizeValue.parseBytesSizeValue;

/**
 * Runs the circuit breaker logic and updates the settings
 */
public class KNNCircuitBreaker {
    public static final String KNN_MEMORY_CIRCUIT_BREAKER_ENABLED = "knn.memory.circuit_breaker.enabled";
    public static final Setting<Boolean> KNN_MEMORY_CIRCUIT_BREAKER_ENABLED_SETTING = Setting.boolSetting(
        KNN_MEMORY_CIRCUIT_BREAKER_ENABLED,
        true,
        NodeScope,
        Dynamic
    );
    public static final String KNN_MEMORY_CIRCUIT_BREAKER_LIMIT = "knn.memory.circuit_breaker.limit";
    public static final String KNN_DEFAULT_MEMORY_CIRCUIT_BREAKER_LIMIT = "50%";
    public static final Setting<ByteSizeValue> KNN_MEMORY_CIRCUIT_BREAKER_LIMIT_SETTING = new Setting<>(
        KNN_MEMORY_CIRCUIT_BREAKER_LIMIT,
        KNN_DEFAULT_MEMORY_CIRCUIT_BREAKER_LIMIT,
        KNNCircuitBreaker::parseByteSizeValue,
        NodeScope,
        Dynamic
    );

    private static ByteSizeValue parseByteSizeValue(String sValue) {
        if (sValue != null && sValue.endsWith("%")) {
            final String percentAsString = sValue.substring(0, sValue.length() - 1);
            try {
                final double percent = Double.parseDouble(percentAsString);
                if (percent < 0 || percent > 100) {
                    throw new OpenSearchParseException("percentage should be in [0-100], got [{}]", percentAsString);
                }
                long physicalMemoryInBytes = OsProbe.getInstance().getTotalPhysicalMemorySize();
                if (physicalMemoryInBytes <= 0) {
                    throw new IllegalStateException("Physical memory size could not be determined");
                }
                long esJvmSizeInBytes = JvmInfo.jvmInfo().getMem().getHeapMax().getBytes();
                long eligibleMemoryInBytes = physicalMemoryInBytes - esJvmSizeInBytes;
                return new ByteSizeValue((long) ((percent / 100) * eligibleMemoryInBytes), ByteSizeUnit.BYTES);
            } catch (NumberFormatException e) {
                throw new OpenSearchParseException("failed to parse [{}] as a double", e, percentAsString);
            }
        } else {
            return parseBytesSizeValue(sValue, KNN_MEMORY_CIRCUIT_BREAKER_LIMIT);
        }
    }

    public static final String KNN_CIRCUIT_BREAKER_TRIGGERED = "knn.circuit_breaker.triggered";
    public static final Setting<Boolean> KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING = Setting.boolSetting(
        KNN_CIRCUIT_BREAKER_TRIGGERED,
        false,
        NodeScope,
        Dynamic
    );

    public static final String KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE = "knn.circuit_breaker.unset.percentage";
    public static final Integer KNN_DEFAULT_CIRCUIT_BREAKER_UNSET_PERCENTAGE = 75;
    public static final Setting<Double> KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING = Setting.doubleSetting(
        KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE,
        KNN_DEFAULT_CIRCUIT_BREAKER_UNSET_PERCENTAGE,
        0,
        100,
        NodeScope,
        Dynamic
    );

    private static Logger logger = LogManager.getLogger(KNNCircuitBreaker.class);
    public static int CB_TIME_INTERVAL = 2 * 60; // seconds

    private static KNNCircuitBreaker INSTANCE;
    private ThreadPool threadPool;
    private ClusterService clusterService;

    private KNNCircuitBreaker() {}

    public static synchronized KNNCircuitBreaker getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new KNNCircuitBreaker();
        }
        return INSTANCE;
    }

    /**
     * SetInstance of Circuit Breaker
     *
     * @param instance KNNCircuitBreaker instance
     */
    public static synchronized void setInstance(KNNCircuitBreaker instance) {
        INSTANCE = instance;
    }

    public void initialize(ThreadPool threadPool, ClusterService clusterService) {
        this.threadPool = threadPool;
        this.clusterService = clusterService;
        NativeMemoryCacheManager nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        Runnable runnable = () -> {
            if (nativeMemoryCacheManager.isCacheCapacityReached() && clusterService.localNode().isDataNode()) {
                long currentSizeKiloBytes = nativeMemoryCacheManager.getCacheSizeInKilobytes();
                long circuitBreakerLimitSizeKiloBytes = KNNCircuitBreakerUtil.instance().getCircuitBreakerLimit().getKb();
                long circuitBreakerUnsetSizeKiloBytes = (long) ((KNNCircuitBreakerUtil.instance().getCircuitBreakerUnsetPercentage() / 100)
                    * circuitBreakerLimitSizeKiloBytes);
                /**
                 * Unset capacityReached flag if currentSizeBytes is less than circuitBreakerUnsetSizeBytes
                 */
                if (currentSizeKiloBytes <= circuitBreakerUnsetSizeKiloBytes) {
                    nativeMemoryCacheManager.setCacheCapacityReached(false);
                }
            }

            // Leader node untriggers CB if all nodes have not reached their max capacity
            if (KNNCircuitBreakerUtil.instance().isCircuitBreakerTriggered()
                && clusterService.state().nodes().isLocalNodeElectedClusterManager()) {
                List<String> nodesAtMaxCapacity;
                try {
                    nodesAtMaxCapacity = KNNCircuitBreakerUtil.instance().getNodesAtMaxCapacity();
                } catch (ExecutionException | InterruptedException e) {
                    logger.error("Unable to get knn stats and determine if any nodes are at capacity", e);
                    return;
                }

                if (!nodesAtMaxCapacity.isEmpty()) {
                    logger.info(
                        "[KNN] knn.circuit_breaker.triggered stays set. Nodes at max cache capacity: "
                            + String.join(",", nodesAtMaxCapacity)
                            + "."
                    );
                } else {
                    logger.info("No nodes are at max cache capacity. Unsetting knn.circuit_breaker.triggered flag.");
                    KNNCircuitBreakerUtil.instance().updateCircuitBreakerSettings(false);
                }
            }
        };
        this.threadPool.scheduleWithFixedDelay(runnable, TimeValue.timeValueSeconds(CB_TIME_INTERVAL), ThreadPool.Names.GENERIC);
    }
}
