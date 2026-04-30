/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.CodecFormatResolver;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.BEAM_WIDTH;
import static org.opensearch.knn.common.KNNConstants.MAX_CONNECTIONS;

/**
 * Standalone registry-based base class for per-field KNN vectors format resolution,
 * used by current and future codecs (KNN1040+).
 *
 * <p>
 * This class extends {@link PerFieldKnnVectorsFormat} directly and delegates format
 * construction to {@link CodecFormatResolver} instances for each engine type.
 * </p>
 */
@Log4j2
public abstract class KNN1040BasePerFieldKnnVectorsFormat extends PerFieldKnnVectorsFormat {

    private final Optional<MapperService> mapperService;
    private final int defaultMaxConnections;
    private final int defaultBeamWidth;
    private final Supplier<KnnVectorsFormat> defaultFormatSupplier;
    private final CodecFormatResolver luceneFormatResolver;
    private final CodecFormatResolver nativeFormatResolver;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    protected KNN1040BasePerFieldKnnVectorsFormat(
        Optional<MapperService> mapperService,
        int defaultMaxConnections,
        int defaultBeamWidth,
        Supplier<KnnVectorsFormat> defaultFormatSupplier,
        CodecFormatResolver luceneFormatResolver,
        CodecFormatResolver nativeFormatResolver,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.mapperService = mapperService;
        this.defaultMaxConnections = defaultMaxConnections;
        this.defaultBeamWidth = defaultBeamWidth;
        this.defaultFormatSupplier = defaultFormatSupplier;
        this.luceneFormatResolver = luceneFormatResolver;
        this.nativeFormatResolver = nativeFormatResolver;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    @Override
    public KnnVectorsFormat getKnnVectorsFormatForField(final String field) {
        if (isKnnVectorFieldType(field) == false) {
            log.debug(
                "Initialize KNN vector format for field [{}] with default params [{}] = \"{}\" and [{}] = \"{}\"",
                field,
                MAX_CONNECTIONS,
                defaultMaxConnections,
                BEAM_WIDTH,
                defaultBeamWidth
            );
            return defaultFormatSupplier.get();
        }
        KNNVectorFieldType mappedFieldType = (KNNVectorFieldType) mapperService.orElseThrow(
            () -> new IllegalStateException(
                String.format("Cannot read field type for field [%s] because mapper service is not available", field)
            )
        ).fieldType(field);

        final KNNMappingConfig knnMappingConfig = mappedFieldType.getKnnMappingConfig();
        if (knnMappingConfig.getModelId().isPresent()) {
            return nativeFormatResolver.resolve();
        }

        final KNNMethodContext knnMethodContext = knnMappingConfig.getKnnMethodContext()
            .orElseThrow(() -> new IllegalArgumentException("KNN method context cannot be empty"));
        nativeIndexBuildStrategyFactory.setKnnLibraryIndexingContext(knnMappingConfig.getKnnLibraryIndexingContext());
        final KNNEngine engine = knnMethodContext.getKnnEngine();
        final Map<String, Object> params = knnMethodContext.getMethodComponentContext().getParameters();

        if (engine == KNNEngine.LUCENE) {
            return luceneFormatResolver.resolve(field, knnMethodContext, params, defaultMaxConnections, defaultBeamWidth);
        }

        // Native engines — pass params so the resolver can detect SQ encoder
        return nativeFormatResolver.resolve(field, knnMethodContext, params, defaultMaxConnections, defaultBeamWidth);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return getKnnVectorsFormatForField(fieldName).getMaxDimensions(fieldName);
    }

    private boolean isKnnVectorFieldType(final String field) {
        return mapperService.isPresent() && mapperService.get().fieldType(field) instanceof KNNVectorFieldType;
    }
}
