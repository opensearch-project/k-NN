/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.grpc.proto.request.search.query;

import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.parser.KNNQueryBuilderParser;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.transport.grpc.proto.request.search.query.QueryBuilderProtoConverterRegistry;
import org.opensearch.protobufs.KnnQuery;
import org.opensearch.protobufs.KnnQueryRescore;
import org.opensearch.protobufs.QueryContainer;

import java.util.List;

/**
 * Utility class for converting KNN Protocol Buffers to OpenSearch objects.
 * This class provides methods to transform Protocol Buffer representations of KNN queries
 * into their corresponding OpenSearch KNNQueryBuilder implementations for search operations.
 */
public class KNNQueryBuilderProtoUtils {

    // Registry for query conversion
    private static QueryBuilderProtoConverterRegistry REGISTRY = new QueryBuilderProtoConverterRegistry();

    private KNNQueryBuilderProtoUtils() {
        // Utility class, no instances
    }

    /**
     * Sets the registry for testing purposes.
     *
     * @param registry The registry to use
     */
    static void setRegistry(QueryBuilderProtoConverterRegistry registry) {
        REGISTRY = registry;
    }

    /**
     * Gets the current registry.
     *
     * @return The current registry
     */
    static QueryBuilderProtoConverterRegistry getRegistry() {
        return REGISTRY;
    }

    /**
     * Converts a Protocol Buffer KnnQuery to an OpenSearch KNNQueryBuilder.
     * Similar to {@link KNNQueryBuilderParser#fromXContent(XContentParser)}, this method
     * parses the Protocol Buffer representation and creates a properly configured
     * KNNQueryBuilder with the appropriate field name, vector, k, and other settings.
     *
     * @param knnQueryProto The Protocol Buffer KnnQuery to convert
     * @return A configured KNNQueryBuilder instance
     */
    public static QueryBuilder fromProto(KnnQuery knnQueryProto) {

        // Create a builder for the KNNQueryBuilder with the field name
        KNNQueryBuilder.Builder builder = KNNQueryBuilder.builder();
        builder.fieldName(knnQueryProto.getField());

        // Set vector
        List<Float> vectorList = knnQueryProto.getVectorList();
        float[] vector = new float[vectorList.size()];
        for (int i = 0; i < vectorList.size(); i++) {
            vector[i] = vectorList.get(i);
        }

        builder.vector(vector);

        // Set k
        builder.k(knnQueryProto.getK());

        // Set filter if present
        if (knnQueryProto.hasFilter()) {
            // Convert the filter QueryContainer to a QueryBuilder
            QueryContainer filterQueryContainer = knnQueryProto.getFilter();
            // Use our instance of the registry to convert the filter
            builder.filter(REGISTRY.fromProto(filterQueryContainer));
        }

        // Set max distance if present
        if (knnQueryProto.hasMaxDistance()) {
            builder.maxDistance(knnQueryProto.getMaxDistance());
        }

        // Set min score if present
        if (knnQueryProto.hasMinScore()) {
            builder.minScore(knnQueryProto.getMinScore());
        }

        // Set method parameters if present
        if (knnQueryProto.hasMethodParameters()) {
            // Convert ObjectMap to Java Map
            java.util.Map<String, Object> methodParameters = new java.util.HashMap<>();
            org.opensearch.protobufs.ObjectMap objectMap = knnQueryProto.getMethodParameters();

            for (java.util.Map.Entry<String, org.opensearch.protobufs.ObjectMap.Value> entry : objectMap.getFieldsMap().entrySet()) {
                String key = entry.getKey();
                org.opensearch.protobufs.ObjectMap.Value value = entry.getValue();

                // Extract the value based on its type
                Object javaValue = null;
                switch (value.getValueCase()) {
                    case INT32:
                        javaValue = value.getInt32();
                        break;
                    case INT64:
                        javaValue = value.getInt64();
                        break;
                    case FLOAT:
                        javaValue = value.getFloat();
                        break;
                    case DOUBLE:
                        javaValue = value.getDouble();
                        break;
                    case STRING:
                        javaValue = value.getString();
                        break;
                    case BOOL:
                        javaValue = value.getBool();
                        break;
                    // Handle other types as needed
                    default:
                        // Skip unsupported types
                        continue;
                }

                methodParameters.put(key, javaValue);
            }

            builder.methodParameters(methodParameters);
        }

        // Set rescore if present
        if (knnQueryProto.hasRescore()) {
            // Convert the rescore KnnQueryRescore to a RescoreContext
            KnnQueryRescore rescoreProto = knnQueryProto.getRescore();

            if (rescoreProto.getKnnQueryRescoreCase() == KnnQueryRescore.KnnQueryRescoreCase.ENABLE) {
                // If enable is true, use default rescore context, otherwise explicitly disable
                if (rescoreProto.getEnable()) {
                    builder.rescoreContext(RescoreContext.getDefault());
                } else {
                    builder.rescoreContext(RescoreContext.EXPLICITLY_DISABLED_RESCORE_CONTEXT);
                }
            } else if (rescoreProto.getKnnQueryRescoreCase() == KnnQueryRescore.KnnQueryRescoreCase.CONTEXT) {
                // If context is provided, create a RescoreContext with the specified oversample factor
                org.opensearch.protobufs.RescoreContext contextProto = rescoreProto.getContext();
                if (contextProto.hasOversampleFactor()) {
                    builder.rescoreContext(RescoreContext.builder().oversampleFactor(contextProto.getOversampleFactor()).build());
                } else {
                    builder.rescoreContext(RescoreContext.getDefault());
                }
            }
        }

        // Set expandNested if present
        if (knnQueryProto.hasExpandNestedDocs()) {
            builder.expandNested(knnQueryProto.getExpandNestedDocs());
        }

        // Set boost if present
        if (knnQueryProto.hasBoost()) {
            builder.boost(knnQueryProto.getBoost());
        }

        // Set name if present
        if (knnQueryProto.hasUnderscoreName()) {
            builder.queryName(knnQueryProto.getUnderscoreName());
        }

        return builder.build();
    }
}
