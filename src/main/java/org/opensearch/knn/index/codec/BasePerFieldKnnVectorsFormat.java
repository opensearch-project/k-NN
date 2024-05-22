/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines99KnnVectorsFormat;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelCache;

import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Supplier;

/**
 * Base class for PerFieldKnnVectorsFormat, builds KnnVectorsFormat based on specific Lucene version
 */
@AllArgsConstructor
@Log4j2
public abstract class BasePerFieldKnnVectorsFormat extends PerFieldKnnVectorsFormat {

    private final Optional<MapperService> optionalMapperService;
    private final int defaultMaxConnections;
    private final int defaultBeamWidth;
    private final Supplier<KnnVectorsFormat> defaultFormatSupplier;
    private final BiFunction<Integer, Integer, KnnVectorsFormat> formatSupplier;

    @Override
    public KnnVectorsFormat getKnnVectorsFormatForField(final String field) {
        if (isKnnVectorFieldType(field) == false) {
            log.debug(
                "Initialize KNN vector format for field [{}] with default params [max_connections] = \"{}\" and [beam_width] = \"{}\"",
                field,
                defaultMaxConnections,
                defaultBeamWidth
            );
            return defaultFormatSupplier.get();
        }
        if (optionalMapperService.isEmpty()) {
            throw new IllegalStateException(
                String.format("Cannot read field type for field [%s] because mapper service is not available", field)
            );
        }
        final KNNVectorFieldMapper.KNNVectorFieldType mappedFieldType = (KNNVectorFieldMapper.KNNVectorFieldType) optionalMapperService
            .get()
            .fieldType(field);

        final KNNEngine knnEngine = getKNNEngine(mappedFieldType);
        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine)) {
            log.debug("Native Engine present hence using NativeEnginesKNNVectorsFormat. Engine found: {}", knnEngine);
            return new NativeEngines99KnnVectorsFormat();
        }

        final Map<String, Object> params = mappedFieldType.getKnnMethodContext().getMethodComponentContext().getParameters();
        int maxConnections = getMaxConnections(params);
        int beamWidth = getBeamWidth(params);
        log.debug(
            "Initialize KNN vector format for field [{}] with params [max_connections] = \"{}\" and [beam_width] = \"{}\"",
            field,
            maxConnections,
            beamWidth
        );
        return formatSupplier.apply(maxConnections, beamWidth);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return getKnnVectorsFormatForField(fieldName).getMaxDimensions(fieldName);
    }

    private boolean isKnnVectorFieldType(final String field) {
        return optionalMapperService.isPresent()
            && optionalMapperService.get().fieldType(field) instanceof KNNVectorFieldMapper.KNNVectorFieldType;
    }

    private int getMaxConnections(final Map<String, Object> params) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_M)) {
            return (int) params.get(KNNConstants.METHOD_PARAMETER_M);
        }
        return defaultMaxConnections;
    }

    private int getBeamWidth(final Map<String, Object> params) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION)) {
            return (int) params.get(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION);
        }
        return defaultBeamWidth;
    }

    private KNNEngine getKNNEngine(final KNNVectorFieldMapper.KNNVectorFieldType mappedFieldType) {
        final String modelId = mappedFieldType.getModelId();
        if (modelId != null) {
            var model = ModelCache.getInstance().get(modelId);
            return model.getModelMetadata().getKnnEngine();
        }

        if (mappedFieldType.getKnnMethodContext() == null) {
            return KNNEngine.DEFAULT;
        } else {
            return mappedFieldType.getKnnMethodContext().getKnnEngine();
        }
    }
}
