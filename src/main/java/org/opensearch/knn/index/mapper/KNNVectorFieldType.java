/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
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
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
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
import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;

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
    Version indexCreatedVersion;
    /**
     * Supplier of the resolved index spec, never null. Engine fields supply an eagerly computed spec;
     * model fields supply a lazy resolver because model metadata lives in cluster state, which may not
     * be available during field mapper creation. The resolved value is memoized in {@link #resolvedSpec}.
     */
    @Getter(AccessLevel.NONE)
    Supplier<ResolvedIndexSpec> resolvedSpecSupplier;
    @Getter(AccessLevel.NONE)
    volatile ResolvedIndexSpec resolvedSpec;

    /**
     * Constructor for KNNVectorFieldType with index created version. The resolved index spec defaults
     * to a lazily constructed no-ANN spec.
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
        this(
            name,
            metadata,
            vectorDataType,
            annConfig,
            indexCreatedVersion,
            () -> ResolvedIndexSpec.noAnn(vectorDataType, annConfig.getDimension(), indexCreatedVersion)
        );
    }

    /**
     * Constructor for KNNVectorFieldType with index created version and resolved index spec.
     *
     * @param name name of the field
     * @param metadata metadata of the field
     * @param vectorDataType data type of the vector
     * @param annConfig configuration context for the ANN index
     * @param indexCreatedVersion Index created version.
     * @param resolvedSpec resolved index spec, must not be null
     */
    public KNNVectorFieldType(
        String name,
        Map<String, String> metadata,
        VectorDataType vectorDataType,
        KNNMappingConfig annConfig,
        Version indexCreatedVersion,
        ResolvedIndexSpec resolvedSpec
    ) {
        this(name, metadata, vectorDataType, annConfig, indexCreatedVersion, () -> resolvedSpec);
        Objects.requireNonNull(resolvedSpec, "resolvedSpec must not be null");
    }

    /**
     * Constructor for KNNVectorFieldType with a lazy resolved index spec supplier.
     *
     * @param name name of the field
     * @param metadata metadata of the field
     * @param vectorDataType data type of the vector
     * @param annConfig configuration context for the ANN index
     * @param indexCreatedVersion Index created version.
     * @param resolvedSpecSupplier supplier of the resolved index spec, must not be null and must not supply null
     */
    public KNNVectorFieldType(
        String name,
        Map<String, String> metadata,
        VectorDataType vectorDataType,
        KNNMappingConfig annConfig,
        Version indexCreatedVersion,
        Supplier<ResolvedIndexSpec> resolvedSpecSupplier
    ) {
        this(name, metadata, vectorDataType, annConfig);
        this.indexCreatedVersion = indexCreatedVersion;
        this.resolvedSpecSupplier = Objects.requireNonNull(resolvedSpecSupplier, "resolvedSpecSupplier must not be null");
    }

    /**
     * Constructor for KNNVectorFieldType. The resolved index spec defaults to a lazily constructed
     * no-ANN spec.
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
        this.resolvedSpecSupplier = () -> ResolvedIndexSpec.noAnn(vectorDataType, annConfig.getDimension(), null);
    }

    /**
     * Returns the resolved index spec for this field, never null. Resolved lazily on first access and
     * memoized; recomputation on a concurrent first access is benign since suppliers are idempotent
     * (same pattern as the lazy model-backed {@link KNNMappingConfig}).
     *
     * @return the resolved index spec
     */
    public ResolvedIndexSpec getResolvedSpec() {
        ResolvedIndexSpec spec = resolvedSpec;
        if (spec == null) {
            spec = Objects.requireNonNull(resolvedSpecSupplier.get(), "resolvedSpecSupplier must not supply a null spec");
            resolvedSpec = spec;
        }
        return spec;
    }

    /**
     * When {@code true}, memory-optimized search is always applied for this field regardless of the
     * cluster-level setting (e.g., FAISS SQ 1-bit always requires memory-optimized search).
     *
     * @see ResolvedIndexSpec#alwaysUseMemoryOptimizedSearch()
     */
    public boolean isAlwaysUseMemoryOptimizedSearch() {
        return getResolvedSpec().alwaysUseMemoryOptimizedSearch();
    }

    /**
     * Whether this field type can benefit from memory-optimized search. A field may be eligible for
     * memory-optimized search but still require the cluster-level setting to be enabled, unless
     * {@link #isAlwaysUseMemoryOptimizedSearch()} is {@code true}.
     *
     * @see ResolvedIndexSpec#isMemoryOptimizedEligible()
     */
    public boolean isMemoryOptimizedSearchAvailable() {
        return getResolvedSpec().isMemoryOptimizedEligible();
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
        return getResolvedSpec().getRescoreContext();
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
}
