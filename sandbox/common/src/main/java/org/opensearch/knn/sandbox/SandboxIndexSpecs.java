/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox;

import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;

/**
 * Builds the minimal {@link ResolvedIndexSpec} a sandbox method attaches to its indexing context.
 *
 * <p>The field mapper requires a spec on every mapped field ({@code KNNVectorFieldType}). Core methods
 * extending {@code AbstractKNNMethod} get theirs built automatically; a method implementing
 * {@code KNNMethod} directly supplies its own, and this helper covers the common case: engine, method
 * name, data type, dimension and index version carried over, every behavioral answer left at the
 * builder's "off" defaults (no radial, no default rescore, no memory-optimized search, no remote build).
 */
public final class SandboxIndexSpecs {

    /**
     * A spec carrying identity and shape only, safe for either context being null (the mapper paths that
     * pass no context still receive a non-null spec).
     */
    public static ResolvedIndexSpec minimalSpec(String methodName, KNNMethodContext methodContext, KNNMethodConfigContext configContext) {
        final ResolvedIndexSpec.ResolvedIndexSpecBuilder builder = ResolvedIndexSpec.builder().methodName(methodName);
        if (methodContext != null) {
            builder.engine(methodContext.getKnnEngine());
        }
        if (configContext != null) {
            builder.vectorDataType(configContext.getVectorDataType());
            if (configContext.getDimension() != null) {
                builder.dimension(configContext.getDimension());
            }
            builder.indexVersionCreated(configContext.getVersionCreated());
        }
        return builder.build();
    }

    private SandboxIndexSpecs() {}
}
