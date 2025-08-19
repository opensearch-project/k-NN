/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.grpc.proto.request.search.query;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.transport.grpc.spi.QueryBuilderProtoConverterRegistry;

/**
 * Utility class to access the gRPC registry from the transport-grpc plugin.
 * This provides a way for k-NN to access the populated registry with built-in converters.
 */
public class GrpcRegistryAccessor {
    private static final Logger logger = LogManager.getLogger(GrpcRegistryAccessor.class);

    private static volatile QueryBuilderProtoConverterRegistry sharedRegistry;

    /**
     * Sets the shared registry instance. This should be called during plugin initialization
     * when the transport-grpc registry becomes available.
     *
     * @param registry The populated registry from transport-grpc plugin
     */
    public static void setSharedRegistry(QueryBuilderProtoConverterRegistry registry) {
        if (registry != null) {
            sharedRegistry = registry;
            logger.info("gRPC registry accessor initialized with shared registry");
        } else {
            logger.warn("Attempted to set null registry in GrpcRegistryAccessor");
        }
    }

    /**
     * Gets the shared registry instance.
     *
     * @return The shared registry, or null if not yet initialized
     */
    public static QueryBuilderProtoConverterRegistry getSharedRegistry() {
        return sharedRegistry;
    }

    /**
     * Checks if the shared registry is available.
     *
     * @return true if the registry is available, false otherwise
     */
    public static boolean isRegistryAvailable() {
        return sharedRegistry != null;
    }
}
