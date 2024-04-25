/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import java.io.IOException;
import java.util.Arrays;

import java.util.List;
import java.util.Objects;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.opensearch.core.common.Strings;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.apache.lucene.search.Query;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryShardContext;

import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNValidationUtil.validateByteVectorValue;
import static org.opensearch.knn.index.IndexUtil.isClusterOnOrAfterMinRequiredVersion;
import static org.opensearch.knn.index.util.KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH;

/**
 * Helper class to build the KNN query
 */
@Log4j2
public class KNNQueryBuilder extends AbstractQueryBuilder<KNNQueryBuilder> {
    private static ModelDao modelDao;

    public static final ParseField VECTOR_FIELD = new ParseField("vector");
    public static final ParseField K_FIELD = new ParseField("k");
    public static final ParseField FILTER_FIELD = new ParseField("filter");
    public static final ParseField IGNORE_UNMAPPED_FIELD = new ParseField("ignore_unmapped");
    public static final ParseField MAX_DISTANCE_FIELD = new ParseField(MAX_DISTANCE);
    public static final ParseField MIN_SCORE_FIELD = new ParseField(MIN_SCORE);
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
    private int k = 0;
    private Float maxDistance = null;
    private Float minScore = null;
    private QueryBuilder filter;
    private boolean ignoreUnmapped = false;

    /**
     * Constructs a new query with the given field name and vector
     *
     * @param fieldName Name of the field
     * @param vector    Array of floating points
     */
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
     * Builder method for k
     *
     * @param k K nearest neighbours for the given vector
     */
    public KNNQueryBuilder k(Integer k) {
        if (k == null) {
            throw new IllegalArgumentException(String.format("[%s] requires k to be set", NAME));
        }
        validateSingleQueryType(k, maxDistance, minScore);
        if (k <= 0 || k > K_MAX) {
            throw new IllegalArgumentException(String.format("[%s] requires k to be in the range (0, %d]", NAME, K_MAX));
        }
        this.k = k;
        return this;
    }

    /**
     * Builder method for maxDistance
     *
     * @param maxDistance the maxDistance threshold for the nearest neighbours
     */
    public KNNQueryBuilder maxDistance(Float maxDistance) {
        if (maxDistance == null) {
            throw new IllegalArgumentException(String.format("[%s] requires maxDistance to be set", NAME));
        }
        validateSingleQueryType(k, maxDistance, minScore);
        this.maxDistance = maxDistance;
        return this;
    }

    /**
     * Builder method for minScore
     *
     * @param minScore the minScore threshold for the nearest neighbours
     */
    public KNNQueryBuilder minScore(Float minScore) {
        if (minScore == null) {
            throw new IllegalArgumentException(String.format("[%s] requires minScore to be set", NAME));
        }
        validateSingleQueryType(k, maxDistance, minScore);
        if (minScore <= 0) {
            throw new IllegalArgumentException(String.format("[%s] requires minScore to be greater than 0", NAME));
        }
        this.minScore = minScore;
        return this;
    }

    /**
     * Builder method for filter
     *
     * @param filter QueryBuilder
     */
    public KNNQueryBuilder filter(QueryBuilder filter) {
        this.filter = filter;
        return this;
    }

    /**
     * Constructs a new query for top k search
     *
     * @param fieldName Name of the filed
     * @param vector    Array of floating points
     * @param k         K nearest neighbours for the given vector
     */
    public KNNQueryBuilder(String fieldName, float[] vector, int k) {
        this(fieldName, vector, k, null);
    }

    public KNNQueryBuilder(String fieldName, float[] vector, int k, QueryBuilder filter) {
        if (Strings.isNullOrEmpty(fieldName)) {
            throw new IllegalArgumentException(String.format("[%s] requires fieldName", NAME));
        }
        if (vector == null) {
            throw new IllegalArgumentException(String.format("[%s] requires query vector", NAME));
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException(String.format("[%s] query vector is empty", NAME));
        }
        if (k <= 0) {
            throw new IllegalArgumentException(String.format("[%s] requires k > 0", NAME));
        }
        if (k > K_MAX) {
            throw new IllegalArgumentException(String.format("[%s] requires k <= %d", NAME, K_MAX));
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

    private static float[] ObjectsToFloats(List<Object> objs) {
        if (Objects.isNull(objs) || objs.isEmpty()) {
            throw new IllegalArgumentException(String.format("[%s] field 'vector' requires to be non-null and non-empty", NAME));
        }
        float[] vec = new float[objs.size()];
        for (int i = 0; i < objs.size(); i++) {
            if ((objs.get(i) instanceof Number) == false) {
                throw new IllegalArgumentException(String.format("[%s] field 'vector' requires to be an array of numbers", NAME));
            }
            vec[i] = ((Number) objs.get(i)).floatValue();
        }
        return vec;
    }

    /**
     * @param in Reads from stream
     * @throws IOException Throws IO Exception
     */
    public KNNQueryBuilder(StreamInput in) throws IOException {
        super(in);
        try {
            fieldName = in.readString();
            vector = in.readFloatArray();
            k = in.readInt();
            filter = in.readOptionalNamedWriteable(QueryBuilder.class);
            if (isClusterOnOrAfterMinRequiredVersion("ignore_unmapped")) {
                ignoreUnmapped = in.readOptionalBoolean();
            }
            if (isClusterOnOrAfterMinRequiredVersion(KNNConstants.RADIAL_SEARCH_KEY)) {
                maxDistance = in.readOptionalFloat();
            }
            if (isClusterOnOrAfterMinRequiredVersion(KNNConstants.RADIAL_SEARCH_KEY)) {
                minScore = in.readOptionalFloat();
            }
        } catch (IOException ex) {
            throw new RuntimeException("[KNN] Unable to create KNNQueryBuilder", ex);
        }
    }

    public static KNNQueryBuilder fromXContent(XContentParser parser) throws IOException {
        String fieldName = null;
        List<Object> vector = null;
        float boost = AbstractQueryBuilder.DEFAULT_BOOST;
        Integer k = null;
        Float maxDistance = null;
        Float minScore = null;
        QueryBuilder filter = null;
        String queryName = null;
        String currentFieldName = null;
        boolean ignoreUnmapped = false;
        XContentParser.Token token;
        KNNCounter.KNN_QUERY_REQUESTS.increment();
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token == XContentParser.Token.START_OBJECT) {
                throwParsingExceptionOnMultipleFields(NAME, parser.getTokenLocation(), fieldName, currentFieldName);
                fieldName = currentFieldName;
                while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
                    if (token == XContentParser.Token.FIELD_NAME) {
                        currentFieldName = parser.currentName();
                    } else if (token.isValue() || token == XContentParser.Token.START_ARRAY) {
                        if (VECTOR_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            vector = parser.list();
                        } else if (AbstractQueryBuilder.BOOST_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            boost = parser.floatValue();
                        } else if (K_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            k = (Integer) NumberFieldMapper.NumberType.INTEGER.parse(parser.objectBytes(), false);
                        } else if (IGNORE_UNMAPPED_FIELD.getPreferredName().equals(currentFieldName)) {
                            if (isClusterOnOrAfterMinRequiredVersion("ignore_unmapped")) {
                                ignoreUnmapped = parser.booleanValue();
                            }
                        } else if (AbstractQueryBuilder.NAME_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            queryName = parser.text();
                        } else if (MAX_DISTANCE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            maxDistance = (Float) NumberFieldMapper.NumberType.FLOAT.parse(parser.objectBytes(), false);
                        } else if (MIN_SCORE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            minScore = (Float) NumberFieldMapper.NumberType.FLOAT.parse(parser.objectBytes(), false);
                        } else {
                            throw new ParsingException(
                                parser.getTokenLocation(),
                                "[" + NAME + "] query does not support [" + currentFieldName + "]"
                            );
                        }
                    } else if (token == XContentParser.Token.START_OBJECT) {
                        String tokenName = parser.currentName();
                        if (FILTER_FIELD.getPreferredName().equals(tokenName)) {
                            log.debug(String.format("Start parsing filter for field [%s]", fieldName));
                            KNNCounter.KNN_QUERY_WITH_FILTER_REQUESTS.increment();
                            filter = parseInnerQueryBuilder(parser);
                        } else {
                            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] unknown token [" + token + "]");
                        }
                    } else {
                        throw new ParsingException(
                            parser.getTokenLocation(),
                            "[" + NAME + "] unknown token [" + token + "] after [" + currentFieldName + "]"
                        );
                    }
                }
            } else {
                throwParsingExceptionOnMultipleFields(NAME, parser.getTokenLocation(), fieldName, parser.currentName());
                fieldName = parser.currentName();
                vector = parser.list();
            }
        }

        validateSingleQueryType(k, maxDistance, minScore);

        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(fieldName, ObjectsToFloats(vector)).filter(filter)
            .ignoreUnmapped(ignoreUnmapped)
            .boost(boost)
            .queryName(queryName);

        if (k != null) {
            knnQueryBuilder.k(k);
        } else if (maxDistance != null) {
            knnQueryBuilder.maxDistance(maxDistance);
        } else if (minScore != null) {
            knnQueryBuilder.minScore(minScore);
        }

        return knnQueryBuilder;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(fieldName);
        out.writeFloatArray(vector);
        out.writeInt(k);
        out.writeOptionalNamedWriteable(filter);
        if (isClusterOnOrAfterMinRequiredVersion("ignore_unmapped")) {
            out.writeOptionalBoolean(ignoreUnmapped);
        }
        if (isClusterOnOrAfterMinRequiredVersion(KNNConstants.RADIAL_SEARCH_KEY)) {
            out.writeOptionalFloat(maxDistance);
        }
        if (isClusterOnOrAfterMinRequiredVersion(KNNConstants.RADIAL_SEARCH_KEY)) {
            out.writeOptionalFloat(minScore);
        }
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

    public int getK() {
        return this.k;
    }

    public float getMaxDistance() {
        return this.maxDistance;
    }

    public float getMinScore() {
        return this.minScore;
    }

    public QueryBuilder getFilter() {
        return this.filter;
    }

    /**
     * Sets whether the query builder should ignore unmapped paths (and run a
     * {@link MatchNoDocsQuery} in place of this query) or throw an exception if
     * the path is unmapped.
     */
    public KNNQueryBuilder ignoreUnmapped(boolean ignoreUnmapped) {
        this.ignoreUnmapped = ignoreUnmapped;
        return this;
    }

    public boolean getIgnoreUnmapped() {
        return this.ignoreUnmapped;
    }

    @Override
    public void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.startObject(fieldName);

        builder.field(VECTOR_FIELD.getPreferredName(), vector);
        builder.field(K_FIELD.getPreferredName(), k);
        if (filter != null) {
            builder.field(FILTER_FIELD.getPreferredName(), filter);
        }
        if (maxDistance != null) {
            builder.field(MAX_DISTANCE_FIELD.getPreferredName(), maxDistance);
        }
        if (ignoreUnmapped) {
            builder.field(IGNORE_UNMAPPED_FIELD.getPreferredName(), ignoreUnmapped);
        }
        if (minScore != null) {
            builder.field(MIN_SCORE_FIELD.getPreferredName(), minScore);
        }
        printBoostAndQueryName(builder);
        builder.endObject();
        builder.endObject();
    }

    @Override
    protected Query doToQuery(QueryShardContext context) {
        MappedFieldType mappedFieldType = context.fieldMapper(this.fieldName);

        if (mappedFieldType == null && ignoreUnmapped) {
            return new MatchNoDocsQuery();
        }

        if (!(mappedFieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType)) {
            throw new IllegalArgumentException(String.format("Field '%s' is not knn_vector type.", this.fieldName));
        }

        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldMapper.KNNVectorFieldType) mappedFieldType;
        int fieldDimension = knnVectorFieldType.getDimension();
        KNNMethodContext knnMethodContext = knnVectorFieldType.getKnnMethodContext();
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        VectorDataType vectorDataType = knnVectorFieldType.getVectorDataType();
        SpaceType spaceType = knnVectorFieldType.getSpaceType();

        if (fieldDimension == -1) {
            if (spaceType != null) {
                throw new IllegalStateException("Space type should be null when the field uses a model");
            }
            // If dimension is not set, the field uses a model and the information needs to be retrieved from there
            ModelMetadata modelMetadata = getModelMetadataForField(knnVectorFieldType);
            fieldDimension = modelMetadata.getDimension();
            knnEngine = modelMetadata.getKnnEngine();
            spaceType = modelMetadata.getSpaceType();
        } else if (knnMethodContext != null) {
            // If the dimension is set but the knnMethodContext is not then the field is using the legacy mapping
            knnEngine = knnMethodContext.getKnnEngine();
            spaceType = knnMethodContext.getSpaceType();
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

        if (fieldDimension != vector.length) {
            throw new IllegalArgumentException(
                String.format("Query vector has invalid dimension: %d. Dimension should be: %d", vector.length, fieldDimension)
            );
        }

        byte[] byteVector = new byte[0];
        if (VectorDataType.BYTE == vectorDataType) {
            byteVector = new byte[vector.length];
            for (int i = 0; i < vector.length; i++) {
                validateByteVectorValue(vector[i]);
                byteVector[i] = (byte) vector[i];
            }
            spaceType.validateVector(byteVector);
        } else {
            spaceType.validateVector(vector);
        }

        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine)
            && filter != null
            && !KNNEngine.getEnginesThatSupportsFilters().contains(knnEngine)) {
            throw new IllegalArgumentException(String.format("Engine [%s] does not support filters", knnEngine));
        }

        String indexName = context.index().getName();

        if (k != 0) {
            KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(indexName)
                .fieldName(this.fieldName)
                .vector(VectorDataType.FLOAT == vectorDataType ? this.vector : null)
                .byteVector(VectorDataType.BYTE == vectorDataType ? byteVector : null)
                .vectorDataType(vectorDataType)
                .k(this.k)
                .filter(this.filter)
                .context(context)
                .build();
            return KNNQueryFactory.create(createQueryRequest);
        }
        if (radius != null) {
            if (!ENGINES_SUPPORTING_RADIAL_SEARCH.contains(knnEngine)) {
                throw new UnsupportedOperationException(String.format("Engine [%s] does not support radial search", knnEngine));
            }
            RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(indexName)
                .fieldName(this.fieldName)
                .vector(VectorDataType.FLOAT == vectorDataType ? this.vector : null)
                .byteVector(VectorDataType.BYTE == vectorDataType ? byteVector : null)
                .vectorDataType(vectorDataType)
                .radius(radius)
                .filter(this.filter)
                .context(context)
                .build();
            return RNNQueryFactory.create(createQueryRequest);
        }
        throw new IllegalArgumentException(String.format("[%s] requires k or distance or score to be set", NAME));
    }

    private ModelMetadata getModelMetadataForField(KNNVectorFieldMapper.KNNVectorFieldType knnVectorField) {
        String modelId = knnVectorField.getModelId();

        if (modelId == null) {
            throw new IllegalArgumentException(String.format("Field '%s' does not have model.", this.fieldName));
        }

        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (!ModelUtil.isModelCreated(modelMetadata)) {
            throw new IllegalArgumentException(String.format("Model ID '%s' is not created.", modelId));
        }
        return modelMetadata;
    }

    @Override
    protected boolean doEquals(KNNQueryBuilder other) {
        return Objects.equals(fieldName, other.fieldName)
            && Arrays.equals(vector, other.vector)
            && Objects.equals(k, other.k)
            && Objects.equals(filter, other.filter)
            && Objects.equals(ignoreUnmapped, other.ignoreUnmapped);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, Arrays.hashCode(vector), k, filter, ignoreUnmapped);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    private static void validateSingleQueryType(Integer k, Float distance, Float score) {
        int countSetFields = 0;

        if (k != null && k != 0) {
            countSetFields++;
        }
        if (distance != null) {
            countSetFields++;
        }
        if (score != null) {
            countSetFields++;
        }

        if (countSetFields != 1) {
            throw new IllegalArgumentException(String.format("[%s] requires exactly one of k, distance or score to be set", NAME));
        }
    }
}
