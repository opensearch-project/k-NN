/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.logging.log4j.util.Strings;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.util.Locale;

/**
 * Class contains the logic to figure out what {@link SpaceType} to use based on configuration
 * details. A user can either provide the {@link SpaceType} via the {@link KNNMethodContext} or through a
 * top level parameter. This class will take care of this resolution logic (as well as if neither are configured) and
 * ensure there are not any contradictions.
 */
public final class SpaceTypeResolver {

    public static final SpaceTypeResolver INSTANCE = new SpaceTypeResolver();

    private SpaceTypeResolver() {}

    /**
     * Resolves space type from configuration details. It is guaranteed not to return either null or
     * {@link SpaceType#UNDEFINED}
     *
     * @param knnMethodContext Method context
     * @param vectorDataType Vectordatatype
     * @param topLevelSpaceTypeString Alternative top-level space type
     * @return {@link SpaceType} for the method
     */
    public SpaceType resolveSpaceType(
        final KNNMethodContext knnMethodContext,
        final VectorDataType vectorDataType,
        final String topLevelSpaceTypeString
    ) {
        SpaceType methodSpaceType = getSpaceTypeFromMethodContext(knnMethodContext);
        SpaceType topLevelSpaceType = getSpaceTypeFromString(topLevelSpaceTypeString);

        if (isSpaceTypeConfigured(methodSpaceType) == false && isSpaceTypeConfigured(topLevelSpaceType) == false) {
            return getSpaceTypeFromVectorDataType(vectorDataType);
        }

        if (isSpaceTypeConfigured(methodSpaceType) == false) {
            return topLevelSpaceType;
        }

        if (isSpaceTypeConfigured(topLevelSpaceType) == false) {
            return methodSpaceType;
        }

        if (methodSpaceType == topLevelSpaceType) {
            return topLevelSpaceType;
        }

        throw new MapperParsingException(
            String.format(
                Locale.ROOT,
                "Cannot specify conflicting space types: \"[%s]\" \"[%s]\"",
                methodSpaceType.getValue(),
                topLevelSpaceType.getValue()
            )
        );
    }

    private SpaceType getSpaceTypeFromMethodContext(final KNNMethodContext knnMethodContext) {
        if (knnMethodContext == null) {
            return SpaceType.UNDEFINED;
        }

        return knnMethodContext.getSpaceType();
    }

    private SpaceType getSpaceTypeFromVectorDataType(final VectorDataType vectorDataType) {
        if (vectorDataType == VectorDataType.BINARY) {
            return SpaceType.DEFAULT_BINARY;
        }
        return SpaceType.DEFAULT;
    }

    private SpaceType getSpaceTypeFromString(final String spaceType) {
        if (Strings.isEmpty(spaceType)) {
            return SpaceType.UNDEFINED;
        }

        return SpaceType.getSpace(spaceType);
    }

    private boolean isSpaceTypeConfigured(final SpaceType spaceType) {
        return spaceType != null && spaceType != SpaceType.UNDEFINED;
    }
}
