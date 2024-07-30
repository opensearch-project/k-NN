/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.StringUtils;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.opensearch.common.ValidationException;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.Strings;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorQueryType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.parser.KNNQueryBuilderParser;
import org.opensearch.knn.index.util.EngineSpecificMethodContext;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.index.util.QueryContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNValidationUtil.validateByteVectorValue;
import static org.opensearch.knn.index.query.parser.MethodParametersParser.validateMethodParameters;
import static org.opensearch.knn.index.util.KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH;
import static org.opensearch.knn.validation.ParameterValidator.validateParameters;

/**
 * Helper class to build the KNN query
 */
// The builder validates the member variables so access to the constructor is prohibited to not accidentally bypass validations
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@Log4j2
public class KNNQueryBuilder extends AbstractQueryBuilder<KNNQueryBuilder> {
    private static ModelDao modelDao;

    public static final ParseField VECTOR_FIELD = new ParseField("vector");
    public static final ParseField K_FIELD = new ParseField("k");
    public static final ParseField FILTER_FIELD = new ParseField("filter");
    public static final ParseField IGNORE_UNMAPPED_FIELD = new ParseField("ignore_unmapped");
    public static final ParseField MAX_DISTANCE_FIELD = new ParseField(MAX_DISTANCE);
    public static final ParseField MIN_SCORE_FIELD = new ParseField(MIN_SCORE);
    public static final ParseField EF_SEARCH_FIELD = new ParseField(METHOD_PARAMETER_EF_SEARCH);
    public static final ParseField NPROBE_FIELD = new ParseField(METHOD_PARAMETER_NPROBES);
    public static final ParseField METHOD_PARAMS_FIELD = new ParseField(METHOD_PARAMETER);

    public static final int K_MAX = 10000;
    /**
     * The name for the knn query
     */
    public static final String NAME = "knn";
    /**
     * The default mode terms are combined in a match query
     */
    private final String fieldName;
    private final float[] vector;
    @Getter
    private int k;
    @Getter
    private Float maxDistance;
    @Getter
    private Float minScore;
    @Getter
    private Map<String, ?> methodParameters;
    @Getter
    private QueryBuilder filter;
    @Getter
    private boolean ignoreUnmapped;

    /**
     * Constructs a new query with the given field name and vector
     *
     * @param fieldName Name of the field
     * @param vector    Array of floating points
     * @deprecated Use {@code {@link KNNQueryBuilder.Builder}} instead
     */
    @Deprecated
    public KNNQueryBuilder(String fieldName, float[] vector) {
        if (Strings.isNullOrEmpty(fieldName)) {
            throw new IllegalArgumentException(String.format("[%s] requires fieldName", NAME));
        }
        if (vector == null) {
            throw new IllegalArgumentException(String.format("[%s] requires query vector", NAME));
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException(String.format("[%s] query vector is empty", NAME));
        }
        this.fieldName = fieldName;
        this.vector = vector;
    }

    /**
     * lombok SuperBuilder annotation requires a builder annotation on parent class to work well
     * {@link AbstractQueryBuilder#boost()} and {@link AbstractQueryBuilder#queryName()} both need to be called
     * A custom builder helps with the calls to the parent class, simultaneously addressing the problem of telescoping
     * constructors in this class.
     */
    public static class Builder {
        private String fieldName;
        private float[] vector;
        private Integer k;
        private Map<String, ?> methodParameters;
        private Float maxDistance;
        private Float minScore;
        private QueryBuilder filter;
        private boolean ignoreUnmapped;
        private String queryName;
        private float boost = DEFAULT_BOOST;

        public Builder() {}

        public Builder fieldName(String fieldName) {
            this.fieldName = fieldName;
            return this;
        }

        public Builder vector(float[] vector) {
            this.vector = vector;
            return this;
        }

        public Builder k(Integer k) {
            this.k = k;
            return this;
        }

        public Builder methodParameters(Map<String, ?> methodParameters) {
            this.methodParameters = methodParameters;
            return this;
        }

        public Builder maxDistance(Float maxDistance) {
            this.maxDistance = maxDistance;
            return this;
        }

        public Builder minScore(Float minScore) {
            this.minScore = minScore;
            return this;
        }

        public Builder ignoreUnmapped(boolean ignoreUnmapped) {
            this.ignoreUnmapped = ignoreUnmapped;
            return this;
        }

        public Builder filter(QueryBuilder filter) {
            this.filter = filter;
            return this;
        }

        public Builder queryName(String queryName) {
            this.queryName = queryName;
            return this;
        }

        public Builder boost(float boost) {
            this.boost = boost;
            return this;
        }

        public KNNQueryBuilder build() {
            validate();
            int k = this.k == null ? 0 : this.k;
            return new KNNQueryBuilder(fieldName, vector, k, maxDistance, minScore, methodParameters, filter, ignoreUnmapped).boost(boost)
                .queryName(queryName);
        }

        private void validate() {
            if (Strings.isNullOrEmpty(fieldName)) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires fieldName", NAME));
            }

            if (vector == null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires query vector", NAME));
            } else if (vector.length == 0) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] query vector is empty", NAME));
            }

            if (k == null && minScore == null && maxDistance == null) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME)
                );
            }

            if ((k != null && maxDistance != null) || (maxDistance != null && minScore != null) || (k != null && minScore != null)) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME)
                );
            }

            if (k != null) {
                if (k <= 0 || k > K_MAX) {
                    final String errorMessage = "[" + NAME + "] requires k to be in the range (0, " + K_MAX + "]";
                    throw new IllegalArgumentException(errorMessage);
                }
            }

            if (minScore != null) {
                if (minScore <= 0) {
                    throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires minScore to be greater than 0", NAME));
                }
            }

            if (methodParameters != null) {
                ValidationException validationException = validateMethodParameters(methodParameters);
                if (validationException != null) {
                    throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "[%s] errors in method parameter [%s]", NAME, validationException.getMessage())
                    );
                }
            }
        }
    }

    public static KNNQueryBuilder.Builder builder() {
        return new KNNQueryBuilder.Builder();
    }

    /**
     * Constructs a new query for top k search
     *
     * @param fieldName Name of the filed
     * @param vector    Array of floating points
     * @param k         K nearest neighbours for the given vector
     */
    @Deprecated
    public KNNQueryBuilder(String fieldName, float[] vector, int k) {
        this(fieldName, vector, k, null);
    }

    @Deprecated
    public KNNQueryBuilder(String fieldName, float[] vector, int k, QueryBuilder filter) {
        if (Strings.isNullOrEmpty(fieldName)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires fieldName", NAME));
        }
        if (vector == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires query vector", NAME));
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] query vector is empty", NAME));
        }
        if (k <= 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires k > 0", NAME));
        }
        if (k > K_MAX) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires k <= %d", NAME, K_MAX));
        }

        this.fieldName = fieldName;
        this.vector = vector;
        this.k = k;
        this.filter = filter;
        this.ignoreUnmapped = false;
        this.maxDistance = null;
        this.minScore = null;
    }

    public static void initialize(ModelDao modelDao) {
        KNNQueryBuilder.modelDao = modelDao;
    }

    /**
     * @param in Reads from stream
     * @throws IOException Throws IO Exception
     */
    public KNNQueryBuilder(StreamInput in) throws IOException {
        super(in);
        KNNQueryBuilder.Builder builder = KNNQueryBuilderParser.streamInput(in, IndexUtil::isClusterOnOrAfterMinRequiredVersion);
        fieldName = builder.fieldName;
        vector = builder.vector;
        k = builder.k;
        filter = builder.filter;
        ignoreUnmapped = builder.ignoreUnmapped;
        maxDistance = builder.maxDistance;
        minScore = builder.minScore;
        methodParameters = builder.methodParameters;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        KNNQueryBuilderParser.streamOutput(out, this, IndexUtil::isClusterOnOrAfterMinRequiredVersion);
    }

    /**
     * @return The field name used in this query
     */
    public String fieldName() {
        return this.fieldName;
    }

    /**
     * @return Returns the vector used in this query.
     */
    public Object vector() {
        return this.vector;
    }

    @Override
    public void doXContent(XContentBuilder builder, Params params) throws IOException {
        KNNQueryBuilderParser.toXContent(builder, params, this);
    }

    @Override
    protected Query doToQuery(QueryShardContext context) {
        MappedFieldType mappedFieldType = context.fieldMapper(this.fieldName);

        if (mappedFieldType == null && ignoreUnmapped) {
            return new MatchNoDocsQuery();
        }

        if (!(mappedFieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' is not knn_vector type.", this.fieldName));
        }

        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldMapper.KNNVectorFieldType) mappedFieldType;
        int fieldDimension = knnVectorFieldType.getDimension();
        KNNMethodContext knnMethodContext = knnVectorFieldType.getKnnMethodContext();
        MethodComponentContext methodComponentContext = null;
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        VectorDataType vectorDataType = knnVectorFieldType.getVectorDataType();
        SpaceType spaceType = knnVectorFieldType.getSpaceType();
        VectorQueryType vectorQueryType = getVectorQueryType(k, maxDistance, minScore);
        updateQueryStats(vectorQueryType);

        if (fieldDimension == -1) {
            if (spaceType != null) {
                throw new IllegalStateException("Space type should be null when the field uses a model");
            }
            // If dimension is not set, the field uses a model and the information needs to be retrieved from there
            ModelMetadata modelMetadata = getModelMetadataForField(knnVectorFieldType);
            fieldDimension = modelMetadata.getDimension();
            knnEngine = modelMetadata.getKnnEngine();
            spaceType = modelMetadata.getSpaceType();
            methodComponentContext = modelMetadata.getMethodComponentContext();
            vectorDataType = modelMetadata.getVectorDataType();

        } else if (knnMethodContext != null) {
            // If the dimension is set but the knnMethodContext is not then the field is using the legacy mapping
            knnEngine = knnMethodContext.getKnnEngine();
            spaceType = knnMethodContext.getSpaceType();
            methodComponentContext = knnMethodContext.getMethodComponentContext();
        }

        final String method = methodComponentContext != null ? methodComponentContext.getName() : null;
        if (StringUtils.isNotBlank(method)) {
            final EngineSpecificMethodContext engineSpecificMethodContext = knnEngine.getMethodContext(method);
            QueryContext queryContext = new QueryContext(vectorQueryType);
            ValidationException validationException = validateParameters(
                engineSpecificMethodContext.supportedMethodParameters(queryContext),
                (Map<String, Object>) methodParameters
            );
            if (validationException != null) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Parameters not valid for [%s]:[%s]:[%s] combination: [%s]",
                        knnEngine,
                        method,
                        vectorQueryType.getQueryTypeName(),
                        validationException.getMessage()
                    )
                );
            }
        }

        if (this.maxDistance != null || this.minScore != null) {
            if (!ENGINES_SUPPORTING_RADIAL_SEARCH.contains(knnEngine)) {
                throw new UnsupportedOperationException(
                    String.format(Locale.ROOT, "Engine [%s] does not support radial search", knnEngine)
                );
            }
            if (vectorDataType == VectorDataType.BINARY) {
                throw new UnsupportedOperationException(String.format(Locale.ROOT, "Binary data type does not support radial search"));
            }
        }

        // Currently, k-NN supports distance and score types radial search
        // We need transform distance/score to right type of engine required radius.
        Float radius = null;
        if (this.maxDistance != null) {
            if (this.maxDistance < 0 && SpaceType.INNER_PRODUCT.equals(spaceType) == false) {
                throw new IllegalArgumentException(
                    String.format("[" + NAME + "] requires distance to be non-negative for space type: %s", spaceType)
                );
            }
            radius = knnEngine.distanceToRadialThreshold(this.maxDistance, spaceType);
        }

        if (this.minScore != null) {
            if (this.minScore > 1 && SpaceType.INNER_PRODUCT.equals(spaceType) == false) {
                throw new IllegalArgumentException(
                    String.format("[" + NAME + "] requires score to be in the range [0, 1] for space type: %s", spaceType)
                );
            }
            radius = knnEngine.scoreToRadialThreshold(this.minScore, spaceType);
        }

        int vectorLength = VectorDataType.BINARY == vectorDataType ? vector.length * Byte.SIZE : vector.length;
        if (fieldDimension != vectorLength) {
            throw new IllegalArgumentException(
                String.format("Query vector has invalid dimension: %d. Dimension should be: %d", vectorLength, fieldDimension)
            );
        }

        byte[] byteVector = new byte[0];
        if (VectorDataType.BINARY == vectorDataType) {
            byteVector = new byte[vector.length];
            for (int i = 0; i < vector.length; i++) {
                validateByteVectorValue(vector[i], knnVectorFieldType.getVectorDataType());
                byteVector[i] = (byte) vector[i];
            }
            spaceType.validateVector(byteVector);
        } else if (VectorDataType.BYTE == vectorDataType) {
            byteVector = new byte[vector.length];
            for (int i = 0; i < vector.length; i++) {
                validateByteVectorValue(vector[i], knnVectorFieldType.getVectorDataType());
                byteVector[i] = (byte) vector[i];
            }
            spaceType.validateVector(byteVector);
        } else {
            spaceType.validateVector(vector);
        }

        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine)
            && filter != null
            && !KNNEngine.getEnginesThatSupportsFilters().contains(knnEngine)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Engine [%s] does not support filters", knnEngine));
        }

        String indexName = context.index().getName();

        if (k != 0) {
            KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(indexName)
                .fieldName(this.fieldName)
                .vector(VectorDataType.FLOAT == vectorDataType ? this.vector : null)
                .byteVector(VectorDataType.BYTE == vectorDataType || VectorDataType.BINARY == vectorDataType ? byteVector : null)
                .vectorDataType(vectorDataType)
                .k(this.k)
                .methodParameters(this.methodParameters)
                .filter(this.filter)
                .context(context)
                .build();
            return KNNQueryFactory.create(createQueryRequest);
        }
        if (radius != null) {
            RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(indexName)
                .fieldName(this.fieldName)
                .vector(VectorDataType.FLOAT == vectorDataType ? this.vector : null)
                .byteVector(VectorDataType.BYTE == vectorDataType ? byteVector : null)
                .vectorDataType(vectorDataType)
                .radius(radius)
                .methodParameters(this.methodParameters)
                .filter(this.filter)
                .context(context)
                .build();
            return RNNQueryFactory.create(createQueryRequest);
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires k or distance or score to be set", NAME));
    }

    private ModelMetadata getModelMetadataForField(KNNVectorFieldMapper.KNNVectorFieldType knnVectorField) {
        String modelId = knnVectorField.getModelId();

        if (modelId == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' does not have model.", this.fieldName));
        }

        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (!ModelUtil.isModelCreated(modelMetadata)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Model ID '%s' is not created.", modelId));
        }
        return modelMetadata;
    }

    /**
     * Function to get the vector query type based on the valid query parameter.
     *
     * @param k K nearest neighbours for the given vector, if k is set, then the query type is K
     * @param maxDistance Maximum distance for the given vector, if maxDistance is set, then the query type is MAX_DISTANCE
     * @param minScore Minimum score for the given vector, if minScore is set, then the query type is MIN_SCORE
     */
    private VectorQueryType getVectorQueryType(int k, Float maxDistance, Float minScore) {
        if (maxDistance != null) {
            return VectorQueryType.MAX_DISTANCE;
        }
        if (minScore != null) {
            return VectorQueryType.MIN_SCORE;
        }
        if (k != 0) {
            return VectorQueryType.K;
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME));
    }

    /**
     * Function to update query stats.
     *
     * @param vectorQueryType The type of query to be executed
     */
    private void updateQueryStats(VectorQueryType vectorQueryType) {
        vectorQueryType.getQueryStatCounter().increment();
        if (filter != null) {
            vectorQueryType.getQueryWithFilterStatCounter().increment();
        }
    }

    @Override
    protected boolean doEquals(KNNQueryBuilder other) {
        return Objects.equals(fieldName, other.fieldName)
            && Arrays.equals(vector, other.vector)
            && Objects.equals(k, other.k)
            && Objects.equals(minScore, other.minScore)
            && Objects.equals(maxDistance, other.maxDistance)
            && Objects.equals(methodParameters, other.methodParameters)
            && Objects.equals(filter, other.filter)
            && Objects.equals(ignoreUnmapped, other.ignoreUnmapped);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, Arrays.hashCode(vector), k, methodParameters, filter, ignoreUnmapped, maxDistance, minScore);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    protected QueryBuilder doRewrite(QueryRewriteContext queryShardContext) throws IOException {
        // rewrite filter query if it exists to avoid runtime errors in next steps of query phase
        if (Objects.nonNull(filter)) {
            filter = filter.rewrite(queryShardContext);
        }
        return super.doRewrite(queryShardContext);
    }
}
