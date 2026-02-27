/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.params.KNNBBQVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Base class for PerFieldKnnVectorsFormat, builds KnnVectorsFormat based on specific Lucene version
 */
@AllArgsConstructor
@Log4j2
public abstract class BasePerFieldKnnVectorsFormat extends PerFieldKnnVectorsFormat {

    private final Optional<MapperService> mapperService;
    private final int defaultMaxConnections;
    private final int defaultBeamWidth;
    private final Supplier<KnnVectorsFormat> defaultFormatSupplier;
    private final Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier;
    private final Function<KNNScalarQuantizedVectorsFormatParams, KnnVectorsFormat> scalarQuantizedVectorsFormatSupplier;
    private final Function<KNNBBQVectorsFormatParams, KnnVectorsFormat> bbqVectorsFormatSupplier;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;
    private static final String MAX_CONNECTIONS = "max_connections";
    private static final String BEAM_WIDTH = "beam_width";

    public BasePerFieldKnnVectorsFormat(
        Optional<MapperService> mapperService,
        int defaultMaxConnections,
        int defaultBeamWidth,
        Supplier<KnnVectorsFormat> defaultFormatSupplier,
        Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier
    ) {
        this(mapperService, defaultMaxConnections, defaultBeamWidth, defaultFormatSupplier, vectorsFormatSupplier, null, null);
    }

    public BasePerFieldKnnVectorsFormat(
        Optional<MapperService> mapperService,
        int defaultMaxConnections,
        int defaultBeamWidth,
        Supplier<KnnVectorsFormat> defaultFormatSupplier,
        Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier,
        Function<KNNScalarQuantizedVectorsFormatParams, KnnVectorsFormat> scalarQuantizedVectorsFormatSupplier,
        Function<KNNBBQVectorsFormatParams, KnnVectorsFormat> bbqVectorsFormatSupplier
    ) {
        this(
            mapperService,
            defaultMaxConnections,
            defaultBeamWidth,
            defaultFormatSupplier,
            vectorsFormatSupplier,
            scalarQuantizedVectorsFormatSupplier,
            bbqVectorsFormatSupplier,
            new NativeIndexBuildStrategyFactory()
        );
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
            return nativeEngineVectorsFormat();
        }

        final KNNMethodContext knnMethodContext = knnMappingConfig.getKnnMethodContext()
            .orElseThrow(() -> new IllegalArgumentException("KNN method context cannot be empty"));
        nativeIndexBuildStrategyFactory.setKnnLibraryIndexingContext(knnMappingConfig.getKnnLibraryIndexingContext());
        final KNNEngine engine = knnMethodContext.getKnnEngine();
        final Map<String, Object> params = knnMethodContext.getMethodComponentContext().getParameters();

        if (engine == KNNEngine.LUCENE) {
            if (params != null && params.containsKey(METHOD_ENCODER_PARAMETER)) {
                KNNBBQVectorsFormatParams bbqParams = new KNNBBQVectorsFormatParams(params, defaultMaxConnections, defaultBeamWidth);
                if (bbqParams.validate(params)) {
                    log.debug(
                        "Initialize KNN vector format for field [{}] with binary quantization, params [{}] = \"{}\", [{}] = \"{}\"",
                        field,
                        MAX_CONNECTIONS,
                        bbqParams.getMaxConnections(),
                        BEAM_WIDTH,
                        bbqParams.getBeamWidth()
                    );
                    return bbqVectorsFormatSupplier.apply(bbqParams);
                }

                KNNScalarQuantizedVectorsFormatParams knnScalarQuantizedVectorsFormatParams = new KNNScalarQuantizedVectorsFormatParams(
                    params,
                    defaultMaxConnections,
                    defaultBeamWidth
                );
                if (knnScalarQuantizedVectorsFormatParams.validate(params)) {
                    log.debug(
                        "Initialize KNN vector format for field [{}] with scalar quantization, params [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\"",
                        field,
                        MAX_CONNECTIONS,
                        knnScalarQuantizedVectorsFormatParams.getMaxConnections(),
                        BEAM_WIDTH,
                        knnScalarQuantizedVectorsFormatParams.getBeamWidth(),
                        LUCENE_SQ_CONFIDENCE_INTERVAL,
                        knnScalarQuantizedVectorsFormatParams.getConfidenceInterval(),
                        LUCENE_SQ_BITS,
                        knnScalarQuantizedVectorsFormatParams.getBits()
                    );
                    return scalarQuantizedVectorsFormatSupplier.apply(knnScalarQuantizedVectorsFormatParams);
                }
            }

            KNNVectorsFormatParams knnVectorsFormatParams = new KNNVectorsFormatParams(
                params,
                defaultMaxConnections,
                defaultBeamWidth,
                knnMethodContext.getSpaceType()
            );
            log.debug(
                "Initialize KNN vector format for field [{}] with params [{}] = \"{}\" and [{}] = \"{}\"",
                field,
                MAX_CONNECTIONS,
                knnVectorsFormatParams.getMaxConnections(),
                BEAM_WIDTH,
                knnVectorsFormatParams.getBeamWidth()
            );
            return vectorsFormatSupplier.apply(knnVectorsFormatParams);
        }

        // All native engines to use NativeEngines990KnnVectorsFormat
        return nativeEngineVectorsFormat();
    }

    private NativeEngines990KnnVectorsFormat nativeEngineVectorsFormat() {
        // mapperService is already checked for null or valid instance type at caller, hence we don't need
        // addition isPresent check here.
        final int approximateThreshold = getApproximateThresholdValue();
        return new NativeEngines990KnnVectorsFormat(
            new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer()),
            approximateThreshold,
            nativeIndexBuildStrategyFactory
        );
    }

    private int getApproximateThresholdValue() {
        // This is private method and mapperService is already checked for null or valid instance type before this call
        // at caller, hence we don't need additional isPresent check here.
        final IndexSettings indexSettings = mapperService.get().getIndexSettings();
        final Integer approximateThresholdValue = indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING);
        return approximateThresholdValue != null
            ? approximateThresholdValue
            : KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return getKnnVectorsFormatForField(fieldName).getMaxDimensions(fieldName);
    }

    private boolean isKnnVectorFieldType(final String field) {
        return mapperService.isPresent() && mapperService.get().fieldType(field) instanceof KNNVectorFieldType;
    }
}
