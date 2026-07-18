/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.opensearch.Version;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.ArraySourceValueFetcher;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.QueryShardException;
import org.opensearch.knn.index.KNNVectorDocValueFormat;
import org.opensearch.knn.index.KNNVectorIndexFieldData;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.aggregations.support.CoreValuesSourceType;
import org.opensearch.search.lookup.SearchLookup;

import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.index.engine.KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.deserializeStoredVector;

/**
 * A KNNVector field type to represent the vector field in Opensearch
 */
@Getter
@Log4j2
public class KNNVectorFieldType extends MappedFieldType {
    private static final Logger logger = LogManager.getLogger(KNNVectorFieldType.class);
    KNNMappingConfig knnMappingConfig;
    VectorDataType vectorDataType;
    /**
     * When {@code true}, memory-optimized search is always applied for this field regardless of the
     * cluster-level setting. This is determined at mapping time based on the encoder type
     * (e.g., FAISS SQ encoder always requires memory-optimized search).
     *
     * @see MemoryOptimizedSearchSupportSpec#isAlwaysUseMemoryOptimizedSearch(java.util.Optional)
     */
    boolean alwaysUseMemoryOptimizedSearch;
    /**
     * Whether this field type can benefit from memory-optimized search. This is determined at mapping time
     * based on the engine, method, encoder, and quantization configuration. A field may be eligible for
     * memory-optimized search but still require the cluster-level setting to be enabled, unless
     * {@link #alwaysUseMemoryOptimizedSearch} is {@code true}.
     *
     * @see MemoryOptimizedSearchSupportSpec#isSupportedFieldType(java.util.Optional,
     *      org.opensearch.knn.index.engine.qframe.QuantizationConfig, java.util.Optional)
     */
    boolean memoryOptimizedSearchAvailable;
    Version indexCreatedVersion;

    /**
     * Constructor for KNNVectorFieldType with index created version.
     *
     * @param name name of the field
     * @param metadata metadata of the field
     * @param vectorDataType data type of the vector
     * @param annConfig configuration context for the ANN index
     * @param indexCreatedVersion Index created version.
     */
    public KNNVectorFieldType(
        String name,
        Map<String, String> metadata,
        VectorDataType vectorDataType,
        KNNMappingConfig annConfig,
        Version indexCreatedVersion
    ) {
        this(name, metadata, vectorDataType, annConfig);
        this.alwaysUseMemoryOptimizedSearch = MemoryOptimizedSearchSupportSpec.isAlwaysUseMemoryOptimizedSearch(
            knnMappingConfig.getKnnMethodContext()
        );
        this.memoryOptimizedSearchAvailable = MemoryOptimizedSearchSupportSpec.isSupportedFieldType(
            knnMappingConfig.getKnnMethodContext(),
            annConfig.getQuantizationConfig(),
            annConfig.getModelId()
        );
        this.indexCreatedVersion = indexCreatedVersion;
    }

    /**
     * Constructor for KNNVectorFieldType.
     *
     * @param name name of the field
     * @param metadata metadata of the field
     * @param vectorDataType data type of the vector
     * @param annConfig configuration context for the ANN index
     */
    public KNNVectorFieldType(String name, Map<String, String> metadata, VectorDataType vectorDataType, KNNMappingConfig annConfig) {
        super(name, false, false, true, TextSearchInfo.NONE, metadata);
        this.vectorDataType = vectorDataType;
        this.knnMappingConfig = annConfig;
    }

    @Override
    public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
        return new ArraySourceValueFetcher(name(), context) {
            @Override
            protected Object parseSourceValue(Object value) {
                if (value instanceof ArrayList) {
                    return value;
                } else {
                    logger.warn("Expected type ArrayList for value, but got {} ", value.getClass());
                    return Collections.emptyList();
                }
            }
        };
    }

    @Override
    public String typeName() {
        return KNNVectorFieldMapper.CONTENT_TYPE;
    }

    @Override
    public Query existsQuery(QueryShardContext context) {
        return new FieldExistsQuery(name());
    }

    @Override
    public Query termQuery(Object value, QueryShardContext context) {
        throw new QueryShardException(
            context,
            String.format(Locale.ROOT, "KNN vector do not support exact searching, use KNN queries instead: [%s]", name())
        );
    }

    @Override
    public IndexFieldData.Builder fielddataBuilder(String fullyQualifiedIndexName, Supplier<SearchLookup> searchLookup) {
        failIfNoDocValues();
        return new KNNVectorIndexFieldData.Builder(name(), CoreValuesSourceType.BYTES, this.vectorDataType);
    }

    @Override
    public DocValueFormat docValueFormat(final String format, final ZoneId timeZone) {
        if (timeZone != null) {
            throw new IllegalArgumentException("Field [" + name() + "] of type [" + typeName() + "] does not support custom time zones");
        }
        return KNNVectorDocValueFormat.fromFormatString(format);
    }

    @Override
    public Object valueForDisplay(Object value) {
        return deserializeStoredVector((BytesRef) value, vectorDataType);
    }

    /**
     * Resolve the rescore context provided for a user based on the field configuration
     *
     * @param userProvidedContext {@link RescoreContext} user passed; if null, the default should be configured
     * @return resolved {@link RescoreContext}
     */
    public RescoreContext resolveRescoreContext(RescoreContext userProvidedContext) {
        if (userProvidedContext != null) {
            return userProvidedContext;
        }
        final KNNMappingConfig knnMappingConfig = getKnnMappingConfig();
        final Optional<KNNMethodContext> methodContext = knnMappingConfig.getKnnMethodContext();
        final boolean isFlatMethod = methodContext.isPresent()
            && METHOD_FLAT.equals(methodContext.get().getMethodComponentContext().getName());
        final boolean isSQMultiBit = methodContext.map(mc -> FaissSQEncoder.isSQMultiBit(mc.getMethodComponentContext().getParameters()))
            .orElse(false);
        final int dimension = knnMappingConfig.getDimension();
        final CompressionLevel compressionLevel = knnMappingConfig.getCompressionLevel();
        final Mode mode = knnMappingConfig.getMode();
        KNNEngine engine = null;
        if (methodContext.isPresent()) {
            engine = methodContext.get().getKnnEngine();
        }
        return compressionLevel.getDefaultRescoreContext(
            mode,
            dimension,
            knnMappingConfig.getIndexCreatedVersion(),
            isFlatMethod,
            isSQMultiBit,
            engine
        );
    }

    /**
     * Transforms a query vector based on the field's configuration. The transformation is performed
     * in-place on the input vector according to either the KNN method context or the model ID.
     *
     * @param vector The float array to be transformed in-place. Must not be null.
     * @throws IllegalStateException if neither KNN method context nor Model ID is configured
     *
     * The transformation process follows this order:
     * 1. If vector is not FLOAT type, no transformation is performed
     * 2. Attempts to use KNN method context if present
     * 3. Falls back to model ID if KNN method context is not available
     * 4. Throws exception if neither configuration is present
     */
    public float[] transformQueryVector(float[] vector) {
        if (VectorDataType.FLOAT != vectorDataType) {
            return vector;
        }
        final Optional<KNNMethodContext> knnMethodContext = knnMappingConfig.getKnnMethodContext();
        if (knnMethodContext.isPresent()) {
            KNNMethodContext context = knnMethodContext.get();
            return VectorTransformerFactory.getVectorTransformer(
                context.getKnnEngine(),
                context.getSpaceType(),
                context.getMethodComponentContext()
            ).transform(vector, false);
        }
        final Optional<String> modelId = knnMappingConfig.getModelId();
        if (modelId.isPresent()) {
            ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
            final ModelMetadata metadata = modelDao.getMetadata(modelId.get());
            return VectorTransformerFactory.getVectorTransformer(metadata.getKnnEngine(), metadata.getSpaceType(), null)
                .transform(vector, false);
        }
        throw new IllegalStateException("Either KNN method context or Model Id should be configured");
    }

    /**
     * Validates that the given index configuration supports radial search.
     * Throws {@link UnsupportedOperationException} if radial search is not supported.
     *
     * <p>Radial search is blocked for:</p>
     * <ul>
     *   <li>Engines that do not support radial search (e.g., NMSLIB)</li>
     *   <li>Binary vector data type</li>
     *   <li>BQ (binary quantization) — identified by {@code QuantizationConfig != EMPTY}</li>
     *   <li>Quantized indices that are not 1-bit SQ — among quantized indices, only the
     *       {@code flat} method or the SQ encoder with {@code bits=1} support radial search
     *       via rescoring</li>
     * </ul>
     *
     * <p>Uses the field's own {@code vectorDataType} and {@code knnMappingConfig} for validation.
     * The engine must be passed explicitly because model-based fields resolve their engine via
     * {@code ModelDao}, which is not available on this class.</p>
     *
     * @param knnEngine the engine resolved for this field (from method context or model metadata)
     * @throws UnsupportedOperationException if radial search is not supported
     */
    public void validateSupportRadialSearch(final KNNEngine knnEngine) {
        if (ENGINES_SUPPORTING_RADIAL_SEARCH.contains(knnEngine) == false) {
            throw new UnsupportedOperationException(String.format(Locale.ROOT, "Engine [%s] does not support radial search", knnEngine));
        }
        if (getVectorDataType() == VectorDataType.BINARY) {
            throw new UnsupportedOperationException(String.format(Locale.ROOT, "Binary data type does not support radial search"));
        }
        final KNNMappingConfig mappingConfig = getKnnMappingConfig();
        // BQ (binary quantization) does not support radial search
        if (mappingConfig.getQuantizationConfig() != QuantizationConfig.EMPTY) {
            throw new UnsupportedOperationException("Radial search is not supported for quantized indices using binary quantization.");
        }
        // Among quantized indices, only flat method (32x) or SQ encoder with bits=1
        // support radial search (via rescoring). Non-quantized indices are always allowed.
        final Optional<KNNMethodContext> methodContext = mappingConfig.getKnnMethodContext();
        if (methodContext.isPresent()) {
            // Check compression level first (cheap) before SQ encoder lookup (hash map access)
            final boolean isQuantizedIndex = CompressionLevel.isConfigured(mappingConfig.getCompressionLevel())
                && mappingConfig.getCompressionLevel() != CompressionLevel.x1
                && mappingConfig.getCompressionLevel() != CompressionLevel.x2;
            if (isQuantizedIndex) {
                final boolean isFlatMethod = METHOD_FLAT.equals(methodContext.get().getMethodComponentContext().getName());
                final boolean isSQOneBit = FaissSQEncoder.isSQOneBit(methodContext.get().getMethodComponentContext().getParameters());
                final boolean radialSupported = isFlatMethod || isSQOneBit;
                if (radialSupported == false) {
                    throw new UnsupportedOperationException(
                        "Among quantized indices, radial search is only supported for 1-bit SQ. "
                            + "Current compression level="
                            + mappingConfig.getCompressionLevel()
                            + ", method="
                            + methodContext.get().getMethodComponentContext().getName()
                    );
                }
            }
        }
    }

    /**
     * Determines whether rescoring with full-precision vectors is required after radial search.
     *
     * <p>This method should only be called for knn field types that have already been validated
     * to support radial search via {@link #validateSupportRadialSearch(KNNEngine)}.
     * Calling this on an unsupported configuration may return incorrect results.</p>
     *
     * <p>Currently, only 1-bit SQ (32x compression) requires rescoring, identified by the
     * SQ encoder with {@code bits=1}. Other quantization types may be added in the future.</p>
     *
     * @return {@code true} if rescoring is required after radial search
     */
    public boolean isRescoringRequiredForRadial() {
        final KNNMappingConfig mappingConfig = getKnnMappingConfig();
        final Optional<KNNMethodContext> methodContext = mappingConfig.getKnnMethodContext();
        // Method context is absent for model-based indices, where the field is configured via
        // modelId instead of an explicit method/encoder. Model-based indices do not need rescoring.
        if (methodContext.isPresent() == false) {
            return false;
        }
        // Only 1-bit SQ requires rescoring for radial search to eliminate false positives.
        return FaissSQEncoder.isSQOneBit(methodContext.get().getMethodComponentContext().getParameters());
    }
}
