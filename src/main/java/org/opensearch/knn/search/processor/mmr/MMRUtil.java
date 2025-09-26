/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.opensearch.action.search.SearchRequest;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.plugin.transport.GetModelAction;
import org.opensearch.knn.plugin.transport.GetModelRequest;
import org.opensearch.knn.plugin.transport.GetModelResponse;
import org.opensearch.knn.search.extension.MMRSearchExtBuilder;
import org.opensearch.search.pipeline.ProcessorGenerationContext;
import org.opensearch.transport.client.Client;
import reactor.util.annotation.NonNull;
import reactor.util.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.index.engine.SpaceTypeResolver.getDefaultSpaceType;

/**
 * A util for MMR related functions
 */
public class MMRUtil {

    /**
     * Collect the knn vector field info based on the query field path or the user provided vector field path and
     * the index metadata.
     * @param path The path of the query field or user provided vector field path
     * @param indexMetadataList List of index metadata of the query target indices
     * @return List of knn vector field info
     */
    private static List<MMRVectorFieldInfo> collectKnnVectorFieldInfos(
        @NonNull final String path,
        @NonNull final List<IndexMetadata> indexMetadataList
    ) {
        List<MMRVectorFieldInfo> vectorFieldInfos = new ArrayList<>();

        for (IndexMetadata indexMetadata : indexMetadataList) {
            vectorFieldInfos.add(collectKnnVectorFieldInfo(indexMetadata, path));
        }

        return vectorFieldInfos;
    }

    private static MMRVectorFieldInfo collectKnnVectorFieldInfo(IndexMetadata indexMetadata, String path) {
        final MMRVectorFieldInfo vectorFieldInfo = new MMRVectorFieldInfo();
        vectorFieldInfo.setIndexNameByIndexMetadata(indexMetadata);

        MappingMetadata mappingMetadata = indexMetadata.mapping();
        if (mappingMetadata == null) {
            vectorFieldInfo.setUnmapped(true);
            return vectorFieldInfo;
        }

        Map<String, Object> mapping = mappingMetadata.sourceAsMap();
        Map<String, Object> config = getMMRFieldMappingByPath(mapping, path);
        if (config == null) {
            vectorFieldInfo.setUnmapped(true);
            return vectorFieldInfo;
        }

        vectorFieldInfo.setUnmapped(false);
        vectorFieldInfo.setFieldPath(path);

        String fieldType = (String) config.get(TYPE);
        vectorFieldInfo.setFieldType(fieldType);

        if (KNNVectorFieldMapper.CONTENT_TYPE.equals(fieldType) == false) {
            return vectorFieldInfo; // Not a knn_vector field skip further processing
        }

        vectorFieldInfo.setKnnConfig(config);

        return vectorFieldInfo;
    }

    private static MMRVectorFieldInfo resolveKnnVectorFieldInfo(
        SpaceType userProvidedSpaceType,
        VectorDataType userProvidedVectorDataType,
        List<MMRVectorFieldInfo> MMRVectorFieldInfoList
    ) throws IllegalArgumentException {
        boolean allUnmapped = true;
        List<MMRVectorFieldInfo> nonKnnFields = new ArrayList<>();
        SpaceType resolvedSpaceType = null;
        VectorDataType resolvedVectorDataType = null;

        for (MMRVectorFieldInfo info : MMRVectorFieldInfoList) {
            if (info.isUnmapped()) {
                continue;
            }

            allUnmapped = false;

            if (!info.isKNNVectorField()) {
                nonKnnFields.add(info);
                continue;
            }

            // ensure we have the same space type and vector data type if we have multiple target indices
            resolvedSpaceType = resolveConsistentValue(
                resolvedSpaceType,
                info.getSpaceType(),
                SpaceType::getValue,
                "space type",
                info.getFieldPath()
            );

            resolvedVectorDataType = resolveConsistentValue(
                resolvedVectorDataType,
                info.getVectorDataType(),
                VectorDataType::getValue,
                "vector data type",
                info.getFieldPath()
            );
        }

        if (allUnmapped) {
            resolvedSpaceType = userProvidedSpaceType != null ? userProvidedSpaceType : getDefaultSpaceType(VectorDataType.DEFAULT);
            resolvedVectorDataType = userProvidedVectorDataType != null ? userProvidedVectorDataType : VectorDataType.DEFAULT;
            return new MMRVectorFieldInfo(resolvedSpaceType, resolvedVectorDataType);
        }

        if (!nonKnnFields.isEmpty()) {
            throw new IllegalArgumentException(
                String.format(
                    "MMR query extension cannot support non knn_vector field [%s].",
                    nonKnnFields.stream()
                        .map(info -> String.format(Locale.ROOT, "%s:%s", info.getIndexName(), info.getFieldPath()))
                        .collect(Collectors.joining(","))
                )
            );
        }

        return resolveFinalKnnVectorFieldInfo(userProvidedSpaceType, resolvedSpaceType, userProvidedVectorDataType, resolvedVectorDataType);
    }

    private static <T> T resolveConsistentValue(
        T current,
        T next,
        Function<T, String> valueFormatter,
        String fieldDescription,
        String fieldPath
    ) {
        if (next == null) {
            return current;
        }
        if (current == null) {
            return next;
        }
        if (!current.equals(next)) {
            throw new IllegalArgumentException(
                String.format(
                    "MMR query extension cannot support different %s [%s, %s] for the knn_vector field at path %s.",
                    fieldDescription,
                    valueFormatter.apply(current),
                    valueFormatter.apply(next),
                    fieldPath
                )
            );
        }
        return current;
    }

    private static MMRVectorFieldInfo resolveFinalKnnVectorFieldInfo(
        SpaceType userProvidedSpaceType,
        SpaceType resolvedSpaceType,
        VectorDataType userProvidedVectorDataType,
        VectorDataType resolvedVectorDataType
    ) throws IllegalArgumentException {
        SpaceType finalSpaceType = resolveFinalValue(
            userProvidedSpaceType,
            resolvedSpaceType,
            () -> getDefaultSpaceType(VectorDataType.DEFAULT),
            SpaceType::getValue,
            "space type"
        );

        VectorDataType finalVectorDataType = resolveFinalValue(
            userProvidedVectorDataType,
            resolvedVectorDataType,
            () -> VectorDataType.DEFAULT,
            VectorDataType::getValue,
            "vector data type"
        );

        return new MMRVectorFieldInfo(finalSpaceType, finalVectorDataType);
    }

    private static <T> T resolveFinalValue(
        T userProvided,
        T resolved,
        Supplier<T> defaultSupplier,
        Function<T, String> valueFormatter,
        String fieldDescription
    ) {
        if (userProvided != null && resolved != null && !userProvided.equals(resolved)) {
            throw new IllegalArgumentException(
                String.format(
                    "The %s [%s] provided in the MMR query extension does not match the %s [%s] in target indices.",
                    fieldDescription,
                    valueFormatter.apply(userProvided),
                    fieldDescription,
                    valueFormatter.apply(resolved)
                )
            );
        }

        if (userProvided != null) {
            return userProvided;
        } else if (resolved != null) {
            return resolved;
        } else {
            return defaultSupplier.get();
        }
    }

    private static MMRVectorFieldInfo resolveVectorFieldInfoFromModel(
        VectorDataType userProvidedVectorDataType,
        SpaceType userProvidedSpaceType,
        List<MMRVectorFieldInfo> MMRVectorFieldInfoList,
        Map<String, MMRVectorFieldInfo> modelIdToVectorFieldInfo
    ) throws IllegalArgumentException {
        SpaceType resolvedSpaceType = null;
        VectorDataType resolvedVectorDataType = null;
        for (MMRVectorFieldInfo info : MMRVectorFieldInfoList) {
            SpaceType spaceType;
            VectorDataType vectorDataType;

            // Resolve from model if modelId is present, else from field config
            if (info.getModelId() != null) {
                MMRVectorFieldInfo infoFromModel = modelIdToVectorFieldInfo.get(info.getModelId());
                if (infoFromModel == null) {
                    throw new IllegalStateException(
                        String.format(
                            "Unexpected null when try to resolve the info of the vector field at path [%s] based on its model [%s].",
                            info.getModelId(),
                            info.getFieldPath()
                        )
                    );
                }
                vectorDataType = infoFromModel.getVectorDataType() != null ? infoFromModel.getVectorDataType() : VectorDataType.DEFAULT;
                spaceType = infoFromModel.getSpaceType() != null ? infoFromModel.getSpaceType() : getDefaultSpaceType(vectorDataType);
            } else {
                spaceType = info.getSpaceType();
                vectorDataType = info.getVectorDataType();
            }

            resolvedSpaceType = resolveConsistentValue(
                resolvedSpaceType,
                spaceType,
                SpaceType::getValue,
                "space type",
                info.getFieldPath()
            );

            resolvedVectorDataType = resolveConsistentValue(
                resolvedVectorDataType,
                vectorDataType,
                VectorDataType::getValue,
                "vector data type",
                info.getFieldPath()
            );
        }

        return resolveFinalKnnVectorFieldInfo(userProvidedSpaceType, resolvedSpaceType, userProvidedVectorDataType, resolvedVectorDataType);
    }

    private static void retrieveFieldInfoFromModel(
        @NonNull final Set<String> modelIds,
        @NonNull final Client client,
        @NonNull final ActionListener<Map<String, MMRVectorFieldInfo>> listener
    ) {
        Map<String, MMRVectorFieldInfo> modelIdToVectorFieldInfo = new ConcurrentHashMap<>();
        List<String> errors = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger counter = new AtomicInteger(modelIds.size());

        for (String modelId : modelIds) {
            client.execute(GetModelAction.INSTANCE, new GetModelRequest(modelId), ActionListener.wrap((GetModelResponse response) -> {
                SpaceType spaceTypeFromModel = null;
                VectorDataType vectorDataTypeFromModel = null;
                if (response != null && response.getModel() != null && response.getModel().getModelMetadata() != null) {
                    spaceTypeFromModel = response.getModel().getModelMetadata().getSpaceType();
                    vectorDataTypeFromModel = response.getModel().getModelMetadata().getVectorDataType();
                }
                modelIdToVectorFieldInfo.put(modelId, new MMRVectorFieldInfo(spaceTypeFromModel, vectorDataTypeFromModel));
                if (counter.decrementAndGet() == 0) {
                    listener.onResponse(modelIdToVectorFieldInfo);
                }
            }, (Exception e) -> {
                errors.add(e.getMessage());
                if (counter.decrementAndGet() == 0) {
                    listener.onFailure(
                        new RuntimeException(
                            String.format(
                                Locale.ROOT,
                                "Failed to retrieve model(s) to resolve the space type and vector data type for the MMR query extension. Errors: %s.",
                                String.join(", ", errors)
                            )
                        )
                    );
                }
            }));
        }
    }

    /**
     * Resolves the space type and data type for a vector field, optionally using model metadata if model IDs exist.
     * It will collect the info from the localIndexMetadataList.
     *
     * @param path         the path of the query field or the user provided vector field path
     * @param userProvidedSpaceType   Optional space type provided by the user
     * @param userProvidedVectorDataType Optional vector data type provided by the user
     * @param localIndexMetadataList  List of local index metadata to inspect
     * @param client                  OpenSearch client to fetch models
     * @param continuation            ActionListener callback to receive the resolved MMRVectorFieldInfo
     */
    public static void resolveKnnVectorFieldInfo(
        @NonNull String path,
        @Nullable SpaceType userProvidedSpaceType,
        @Nullable VectorDataType userProvidedVectorDataType,
        @NonNull List<IndexMetadata> localIndexMetadataList,
        @NonNull Client client,
        @NonNull ActionListener<MMRVectorFieldInfo> continuation
    ) {
        try {
            List<MMRVectorFieldInfo> knnVectorFieldInfos = collectKnnVectorFieldInfos(path, localIndexMetadataList);

            resolveKnnVectorFieldInfo(knnVectorFieldInfos, userProvidedSpaceType, userProvidedVectorDataType, client, continuation);
        } catch (Exception e) {
            continuation.onFailure(e);
        }
    }

    /**
     * Resolves the space type and data type for a knn vector field based on field info collected from its index
     * mapping config from multiple indices, optionally using model metadata if model IDs exist.
     *
     * @param MMRVectorFieldInfoList  A list of knn vector info collected from multiple target indices
     * @param userProvidedSpaceType   Optional space type provided by the user
     * @param userProvidedVectorDataType Optional vector data type provided by the user
     * @param client                  OpenSearch client to fetch models
     * @param continuation            callback to execute once the final space type is resolved
     */
    public static void resolveKnnVectorFieldInfo(
        @NonNull List<MMRVectorFieldInfo> MMRVectorFieldInfoList,
        @Nullable SpaceType userProvidedSpaceType,
        @Nullable VectorDataType userProvidedVectorDataType,
        @NonNull Client client,
        @NonNull ActionListener<MMRVectorFieldInfo> continuation
    ) {
        try {
            // Resolve field info based on the field config in index mapping
            MMRVectorFieldInfo resolvedVectorFieldInfo = resolveKnnVectorFieldInfo(
                userProvidedSpaceType,
                userProvidedVectorDataType,
                MMRVectorFieldInfoList
            );

            // Collect model IDs
            Set<String> modelIds = MMRVectorFieldInfoList.stream()
                .map(MMRVectorFieldInfo::getModelId)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());

            if (modelIds.isEmpty()) {
                continuation.onResponse(resolvedVectorFieldInfo);
            } else {
                // Retrieve the field info from the model metadata asynchronously
                retrieveFieldInfoFromModel(modelIds, client, ActionListener.wrap(modelIdToVectorFieldInfo -> {
                    MMRVectorFieldInfo resolvedVectorFieldInfoFromModel = resolveVectorFieldInfoFromModel(
                        userProvidedVectorDataType,
                        userProvidedSpaceType,
                        MMRVectorFieldInfoList,
                        modelIdToVectorFieldInfo
                    );
                    continuation.onResponse(resolvedVectorFieldInfoFromModel);
                }, continuation::onFailure));
            }
        } catch (Exception e) {
            continuation.onFailure(e);
        }
    }

    /**
     * Extracts a dense vector ({@code float[]} or {@code byte[]}) from a document source map given a dot-delimited
     * field path.
     *
     * This utility is designed for KNN / MMR use cases where the vector is expected to be stored
     * as a top-level or single field inside the document. Nested object structures containing
     * vectors are not supported, and will cause an exception.
     *
     * Example:
     * source = Map.of("embedding", List.of(0.1, 0.2, 0.3));
     * float[] vector = VectorUtils.extractVectorFromHit(source, "embedding", "doc-123");
     * vector = [0.1f, 0.2f, 0.3f]
     *
     *
     * @param sourceAsMap The document source returned from {@code hit.getSourceAsMap()}.
     * @param fieldPath   The dot-delimited field path to the vector field (e.g. "embedding" or "nested.field.vector").
     * @param docId       The document ID, used in error messages to help identify problematic documents.
     * @param isFloatVector If the vector is float or byte
     * @return A primitive float/byte array representing the extracted vector.
     */
    @SuppressWarnings("unchecked")
    public static Object extractVectorFromHit(Map<String, Object> sourceAsMap, String fieldPath, String docId, boolean isFloatVector)
        throws IllegalArgumentException {
        String baseError = String.format(Locale.ROOT, "Failed to extract the vector from the doc [%s] for MMR rerank", docId);
        if (sourceAsMap == null || fieldPath == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "%s: source map and fieldPath must not be null.", baseError));
        }

        String[] pathParts = fieldPath.split("\\.");
        Object current = sourceAsMap;

        if (pathParts.length == 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "%s: fieldPath must not be an empty string.", baseError));
        }

        for (int i = 0; i < pathParts.length; i++) {
            String part = pathParts[i];
            if (!(current instanceof Map<?, ?> map)) {
                throw new IllegalArgumentException(
                    String.format("%s: expected object at [%s], but found [%s]", baseError, part, current.getClass().getName())
                );
            }

            current = map.get(part);
            if (current == null) {
                throw new IllegalArgumentException(
                    String.format("%s: field path [%s] not found in document source.", baseError, fieldPath)
                );
            }

            // Final part → should resolve to a vector
            if (i == pathParts.length - 1) {
                if (current instanceof List<?> list) {
                    float[] floatVector = null;
                    byte[] byteVector = null;
                    if (isFloatVector) {
                        floatVector = new float[list.size()];
                    } else {
                        byteVector = new byte[list.size()];
                    }
                    try {
                        for (int j = 0; j < list.size(); j++) {
                            if (isFloatVector) {
                                floatVector[j] = (float) (double) list.get(j);
                            } else {
                                byteVector[j] = (byte) (double) list.get(j);
                            }
                        }
                    } catch (Exception e) {
                        throw new IllegalArgumentException(
                            String.format("%s: unexpected value at the vector field [%s]. error: %s", baseError, fieldPath, e.getMessage())
                        );
                    }
                    if (isFloatVector) {
                        return floatVector;
                    } else return byteVector;
                }
                throw new IllegalArgumentException(
                    String.format(
                        "%s: expected vector (list of numbers) at field path [%s], but found type [%s]",
                        baseError,
                        fieldPath,
                        current.getClass().getName()
                    )
                );
            }
        }

        // Should never reach here
        throw new IllegalStateException(String.format("%s: unexpected error resolving field path [%s].", baseError, fieldPath));
    }

    /**
     * @param processorGenerationContext The context to evaluate if we should generate the MMR processor.
     * @return If the MMR processor should be generated.
     */
    public static boolean shouldGenerateMMRProcessor(ProcessorGenerationContext processorGenerationContext) {
        SearchRequest request = processorGenerationContext.searchRequest();
        if (request == null || request.source() == null || request.source().ext() == null) {
            return false;
        }
        return request.source().ext().stream().anyMatch(MMRSearchExtBuilder.class::isInstance);
    }

    /**
     * Get the field mapping config for a dot-separated path like "user.profile.age" for MMR. The fields on the path
     * should not contain "nested" field type since it means the doc source can have multiple vectors which we cannot
     * support.
     *
     * @param mappings Index mappings
     * @param fieldPath Dot-separated path to the field
     * @return The mapping config map for the field, or null if not found
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> getMMRFieldMappingByPath(Map<String, Object> mappings, @NonNull String fieldPath) {
        if (mappings == null) {
            return null;
        }
        String[] parts = fieldPath.split("\\.");
        Map<String, Object> current = mappings;

        for (int i = 0; i < parts.length; i++) {
            String part = parts[i];
            Object propertiesObj = current.get("properties");
            if (!(propertiesObj instanceof Map)) {
                return null; // no deeper properties
            }

            Map<String, Object> properties = (Map<String, Object>) propertiesObj;
            Object fieldConfig = properties.get(part);
            if (!(fieldConfig instanceof Map)) {
                return null; // field not found
            }
            current = (Map<String, Object>) fieldConfig;

            String fieldType = (String) current.get(TYPE);
            if (ObjectMapper.NESTED_CONTENT_TYPE.equals(fieldType)) {
                throw new IllegalArgumentException(
                    String.format("MMR search extension cannot support the field %s because it is in the nested field %s.", fieldPath, part)
                );
            }
        }
        return current;
    }
}
