/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
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
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

/**
 * Base class for PerFieldKnnVectorsFormat, builds KnnVectorsFormat based on
 * specific Lucene version.
 *
 * <p>
 * Lucene format selection is driven by a registry of format factories keyed by
 * {@link LuceneVectorsFormatType}. Each codec subclass registers the formats it
 * supports.
 * To add a new Lucene format, add an enum value to
 * {@link LuceneVectorsFormatType} and
 * register a factory in the relevant codec subclass(es).
 * </p>
 */
@Log4j2
public abstract class BasePerFieldKnnVectorsFormat extends PerFieldKnnVectorsFormat {

    private final Optional<MapperService> mapperService;
    private final int defaultMaxConnections;
    private final int defaultBeamWidth;
    private final Supplier<KnnVectorsFormat> defaultFormatSupplier;
    private final Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> luceneFormatResolvers;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;
    private static final String MAX_CONNECTIONS = "max_connections";
    private static final String BEAM_WIDTH = "beam_width";

    protected BasePerFieldKnnVectorsFormat(
        Optional<MapperService> mapperService,
        int defaultMaxConnections,
        int defaultBeamWidth,
        Supplier<KnnVectorsFormat> defaultFormatSupplier,
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> luceneFormatResolvers
    ) {
        this(
            mapperService,
            defaultMaxConnections,
            defaultBeamWidth,
            defaultFormatSupplier,
            luceneFormatResolvers,
            new NativeIndexBuildStrategyFactory()
        );
    }

    protected BasePerFieldKnnVectorsFormat(
        Optional<MapperService> mapperService,
        int defaultMaxConnections,
        int defaultBeamWidth,
        Supplier<KnnVectorsFormat> defaultFormatSupplier,
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> luceneFormatResolvers,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.mapperService = mapperService;
        this.defaultMaxConnections = defaultMaxConnections;
        this.defaultBeamWidth = defaultBeamWidth;
        this.defaultFormatSupplier = defaultFormatSupplier;
        this.luceneFormatResolvers = luceneFormatResolvers;
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
            return nativeEngineVectorsFormat();
        }

        final KNNMethodContext knnMethodContext = knnMappingConfig.getKnnMethodContext()
            .orElseThrow(() -> new IllegalArgumentException("KNN method context cannot be empty"));
        nativeIndexBuildStrategyFactory.setKnnLibraryIndexingContext(knnMappingConfig.getKnnLibraryIndexingContext());
        final KNNEngine engine = knnMethodContext.getKnnEngine();
        final Map<String, Object> params = knnMethodContext.getMethodComponentContext().getParameters();

        if (engine == KNNEngine.LUCENE) {
            return resolveLuceneFormat(field, knnMethodContext, params);
        }

        // All native engines to use NativeEngines990KnnVectorsFormat
        return nativeEngineVectorsFormat();
    }

    /**
     * Determines the Lucene format type based on the method context and parameters,
     * then resolves it via the registered format factory.
     */
    private KnnVectorsFormat resolveLuceneFormat(
        final String field,
        final KNNMethodContext methodContext,
        final Map<String, Object> params
    ) {
        final LuceneVectorsFormatType formatType = determineLuceneFormatType(field, methodContext, params);

        final Function<KnnVectorsFormatContext, KnnVectorsFormat> factory = luceneFormatResolvers.get(formatType);
        if (factory == null) {
            throw new IllegalStateException(
                String.format("No Lucene vectors format registered for type [%s] in codec [%s]", formatType, getClass().getSimpleName())
            );
        }

        log.debug("Initialize KNN vector format for field [{}] with Lucene format type [{}]", field, formatType);

        final KnnVectorsFormatContext context = new KnnVectorsFormatContext(
            field,
            methodContext,
            params,
            defaultMaxConnections,
            defaultBeamWidth
        );

        return factory.apply(context);
    }

    /**
     * Routes the method context to the appropriate {@link LuceneVectorsFormatType}.
     */
    private LuceneVectorsFormatType determineLuceneFormatType(
        final String field,
        final KNNMethodContext methodContext,
        final Map<String, Object> params
    ) {
        if (METHOD_FLAT.equals(methodContext.getMethodComponentContext().getName())) {
            log.debug("Initialize KNN vector format for field [{}] with Lucene BBQ flat format", field);
            return LuceneVectorsFormatType.FLAT;
        }

        if (params != null && params.containsKey(METHOD_ENCODER_PARAMETER)) {
            KNNScalarQuantizedVectorsFormatParams sqParams = new KNNScalarQuantizedVectorsFormatParams(
                params,
                defaultMaxConnections,
                defaultBeamWidth
            );
            if (sqParams.validate(params)) {
                log.debug(
                    "Initialize KNN vector format for field [{}] with params [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\"",
                    field,
                    MAX_CONNECTIONS,
                    sqParams.getMaxConnections(),
                    BEAM_WIDTH,
                    sqParams.getBeamWidth(),
                    LUCENE_SQ_CONFIDENCE_INTERVAL,
                    sqParams.getConfidenceInterval(),
                    LUCENE_SQ_BITS,
                    sqParams.getBits()
                );
                return LuceneVectorsFormatType.SCALAR_QUANTIZED;
            }
        }

        KNNVectorsFormatParams hnswParams = new KNNVectorsFormatParams(
            params,
            defaultMaxConnections,
            defaultBeamWidth,
            methodContext.getSpaceType()
        );
        log.debug(
            "Initialize KNN vector format for field [{}] with params [{}] = \"{}\" and [{}] = \"{}\"",
            field,
            MAX_CONNECTIONS,
            hnswParams.getMaxConnections(),
            BEAM_WIDTH,
            hnswParams.getBeamWidth()
        );
        return LuceneVectorsFormatType.HNSW;
    }

    private NativeEngines990KnnVectorsFormat nativeEngineVectorsFormat() {
        // mapperService is already checked for null or valid instance type at caller, hence we don't need
        // addition isPresent check here.
        final int approximateThreshold = getApproximateThresholdValue();
        return new NativeEngines990KnnVectorsFormat(approximateThreshold, nativeIndexBuildStrategyFactory);
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
