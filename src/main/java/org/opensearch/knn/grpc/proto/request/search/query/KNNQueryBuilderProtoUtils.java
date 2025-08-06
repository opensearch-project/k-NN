/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.grpc.proto.request.search.query;

import lombok.experimental.UtilityClass;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.parser.KNNQueryBuilderParser;
import org.opensearch.knn.index.query.request.MethodParameter;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.transport.grpc.proto.request.search.query.QueryBuilderProtoConverterSpiRegistry;
import org.opensearch.protobufs.KnnQuery;
import org.opensearch.protobufs.KnnQueryRescore;
import org.opensearch.protobufs.QueryContainer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility class for converting KNN Protocol Buffers to OpenSearch objects.
 * This class provides methods to transform Protocol Buffer representations of KNN queries
 * into their corresponding OpenSearch KNNQueryBuilder implementations for search operations.
 */
@UtilityClass
public class KNNQueryBuilderProtoUtils {

    // Registry for query conversion
    private static QueryBuilderProtoConverterSpiRegistry REGISTRY = new QueryBuilderProtoConverterSpiRegistry();

    /**
     * Sets the registry for testing purposes.
     *
     * @param registry The registry to use
     */
    void setRegistry(QueryBuilderProtoConverterSpiRegistry registry) {
        REGISTRY = registry;
    }

    /**
     * Gets the current registry.
     *
     * @return The current registry
     */
    QueryBuilderProtoConverterSpiRegistry getRegistry() {
        return REGISTRY;
    }

    /**
    * Converts a Protocol Buffer KnnQuery to an OpenSearch KNNQueryBuilder.
    * This method follows the exact same pattern as {@link KNNQueryBuilderParser#fromXContent(XContentParser)}
    * to ensure parsing consistency and compatibility.
    *
    * @param knnQueryProto The Protocol Buffer KnnQuery to convert
    * @return A configured KNNQueryBuilder instance
    */
    public QueryBuilder fromProto(KnnQuery knnQueryProto) {
        // Create builder using the internal parser pattern like XContent parsing
        KNNQueryBuilder.Builder builder = KNNQueryBuilder.builder();

        // Set field name (equivalent to fieldName parsing in XContent)
        builder.fieldName(knnQueryProto.getField());

        // Set vector (equivalent to VECTOR_FIELD parsing)
        builder.vector(convertVector(knnQueryProto.getVectorList()));

        // Set k if present (equivalent to K_FIELD parsing)
        if (knnQueryProto.getK() > 0) {
            builder.k(knnQueryProto.getK());
        }

        // Set maxDistance if present (equivalent to MAX_DISTANCE_FIELD parsing)
        else if (knnQueryProto.hasMaxDistance()) {
            builder.maxDistance(knnQueryProto.getMaxDistance());
        }

        // Set minScore if present (equivalent to MIN_SCORE_FIELD parsing)
        else if (knnQueryProto.hasMinScore()) {
            builder.minScore(knnQueryProto.getMinScore());
        }

        // Set method parameters (equivalent to METHOD_PARAMS_FIELD parsing)
        if (knnQueryProto.hasMethodParameters()) {
            Map<String, ?> methodParameters = convertMethodParameters(knnQueryProto.getMethodParameters());
            builder.methodParameters(methodParameters);
        }

        // Set filter (equivalent to FILTER_FIELD parsing)
        if (knnQueryProto.hasFilter()) {
            QueryContainer filterQueryContainer = knnQueryProto.getFilter();
            builder.filter(REGISTRY.fromProto(filterQueryContainer));
        }

        // Set rescore (equivalent to RESCORE_FIELD parsing)
        if (knnQueryProto.hasRescore()) {
            RescoreContext rescoreContext = convertRescoreContext(knnQueryProto.getRescore());
            builder.rescoreContext(rescoreContext);
        }

        // Set boost (equivalent to BOOST_FIELD parsing)
        if (knnQueryProto.hasBoost()) {
            builder.boost(knnQueryProto.getBoost());
        }

        // Set query name (equivalent to NAME_FIELD parsing)
        if (knnQueryProto.hasUnderscoreName()) {
            builder.queryName(knnQueryProto.getUnderscoreName());
        }

        // Set expandNested (equivalent to EXPAND_NESTED_FIELD parsing)
        if (knnQueryProto.hasExpandNestedDocs()) {
            builder.expandNested(knnQueryProto.getExpandNestedDocs());
        }

        return builder.build();
    }

    /**
     * Converts a Protocol Buffer vector list to a float array.
     *
     * @param vectorList The Protocol Buffer vector list
     * @return The converted float array
     */
    private float[] convertVector(List<Float> vectorList) {
        float[] vector = new float[vectorList.size()];
        for (int i = 0; i < vectorList.size(); i++) {
            vector[i] = vectorList.get(i);
        }
        return vector;
    }

    /**
     * Converts Protocol Buffer method parameters following the exact same pattern as
     * {@link MethodParametersParser#fromXContent(XContentParser)} to ensure consistency.
     *
     * @param objectMap The Protocol Buffer ObjectMap to convert
     * @return The converted method parameters Map
     */
    private Map<String, ?> convertMethodParameters(org.opensearch.protobufs.ObjectMap objectMap) {
        // First convert Protocol Buffer to raw Map (equivalent to parser.map())
        Map<String, Object> rawMethodParameters = new HashMap<>();
        for (Map.Entry<String, org.opensearch.protobufs.ObjectMap.Value> entry : objectMap.getFieldsMap().entrySet()) {
            String key = entry.getKey();
            Object value = convertObjectMapValue(entry.getValue());
            if (value != null) {
                rawMethodParameters.put(key, value);
            }
        }

        // Then process through MethodParameter.parse() exactly like XContent parsing does
        Map<String, Object> processedMethodParameters = new HashMap<>();
        for (Map.Entry<String, Object> entry : rawMethodParameters.entrySet()) {
            String name = entry.getKey();
            Object value = entry.getValue();

            // Find the MethodParameter enum (same as XContent parsing)
            MethodParameter parameter = MethodParameter.enumOf(name);
            if (parameter == null) {
                throw new IllegalArgumentException("unknown method parameter found [" + name + "]");
            }

            try {
                // Parse using MethodParameter.parse() - this handles type conversion properly
                Object parsedValue = parameter.parse(value);
                processedMethodParameters.put(name, parsedValue);
            } catch (Exception exception) {
                throw new IllegalArgumentException("Error parsing method parameter [" + name + "]: " + exception.getMessage());
            }
        }

        return processedMethodParameters.isEmpty() ? null : processedMethodParameters;
    }

    /**
    * Converts a Protocol Buffer ObjectMap.Value to a Java Object.
    *
    * @param value The Protocol Buffer Value to convert
    * @return The converted Java Object, or null if unsupported type
    */
    private Object convertObjectMapValue(org.opensearch.protobufs.ObjectMap.Value value) {
        switch (value.getValueCase()) {
            case INT32:
                return value.getInt32();
            case INT64:
                return value.getInt64();
            case FLOAT:
                return value.getFloat();
            case DOUBLE:
                return value.getDouble();
            case STRING:
                return value.getString();
            case BOOL:
                return value.getBool();
            default:
                // Skip unsupported types
                return null;
        }
    }

    /**
     * Converts a Protocol Buffer KnnQueryRescore to a RescoreContext.
     *
     * @param rescoreProto The Protocol Buffer KnnQueryRescore to convert
     * @return The converted RescoreContext
     */
    private RescoreContext convertRescoreContext(KnnQueryRescore rescoreProto) {
        switch (rescoreProto.getKnnQueryRescoreCase()) {
            case ENABLE:
                return rescoreProto.getEnable() ? RescoreContext.getDefault() : RescoreContext.EXPLICITLY_DISABLED_RESCORE_CONTEXT;

            case CONTEXT:
                org.opensearch.protobufs.RescoreContext contextProto = rescoreProto.getContext();
                return contextProto.hasOversampleFactor()
                    ? RescoreContext.builder().oversampleFactor(contextProto.getOversampleFactor()).build()
                    : RescoreContext.getDefault();

            default:
                return RescoreContext.getDefault();
        }
    }

}
