/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import java.util.Arrays;

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

import java.io.IOException;
import java.util.List;
import java.util.Objects;

import static org.opensearch.knn.index.IndexUtil.isClusterOnOrAfterMinRequiredVersion;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateByteVectorValue;

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
    public static final ParseField DISTANCE_FIELD = new ParseField("distance");
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
    private Float distance = null;
    private QueryBuilder filter;
    private boolean ignoreUnmapped = false;

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
            throw new IllegalArgumentException("[" + NAME + "] requires fieldName");
        }
        if (vector == null) {
            throw new IllegalArgumentException("[" + NAME + "] requires query vector");
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException("[" + NAME + "] query vector is empty");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("[" + NAME + "] requires k > 0");
        }
        if (k > K_MAX) {
            throw new IllegalArgumentException("[" + NAME + "] requires k <= " + K_MAX);
        }

        this.fieldName = fieldName;
        this.vector = vector;
        this.k = k;
        this.filter = filter;
        this.ignoreUnmapped = false;
        this.distance = null;
    }

    /**
     * Constructs a new query for k-nearest neighbours search or radius-nearest neighbours search with builder
     *
     * @param builder builder to create KNNQueryBuilder
     */
    public KNNQueryBuilder(Builder builder) {
        this.fieldName = builder.fieldName;
        this.vector = builder.vector;
        this.k = builder.k != null ? builder.k : 0;
        this.filter = builder.filter;
        this.ignoreUnmapped = builder.ignoreUnmapped;
        this.distance = builder.distance;
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
                distance = in.readOptionalFloat();
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
        Float distance = null;
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
                        } else if (DISTANCE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            distance = (Float) NumberFieldMapper.NumberType.FLOAT.parse(parser.objectBytes(), false);
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

        return new Builder(fieldName, ObjectsToFloats(vector)).k(k)
            .distance(distance)
            .filter(filter)
            .ignoreUnmapped(ignoreUnmapped)
            .queryName(queryName)
            .boost(boost)
            .build();
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
            out.writeOptionalFloat(distance);
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

    public float getDistance() {
        return this.distance;
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
        if (distance != null) {
            builder.field(DISTANCE_FIELD.getPreferredName(), distance);
        }
        if (ignoreUnmapped) {
            builder.field(IGNORE_UNMAPPED_FIELD.getPreferredName(), ignoreUnmapped);
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
        SpaceType spaceType = SpaceType.DEFAULT;
        VectorDataType vectorDataType = knnVectorFieldType.getVectorDataType();

        if (fieldDimension == -1) {
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

        // Currently, k-NN supports distance type radius search.
        // We need transform distance radius to right type of engine required radius.
        Float engineRadius = null;
        if (this.distance != null) {
            engineRadius = knnEngine.distanceToRadialThreshold(this.distance, spaceType);
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
        }

        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine)
            && filter != null
            && !KNNEngine.getEnginesThatSupportsFilters().contains(knnEngine)) {
            throw new IllegalArgumentException(String.format("Engine [%s] does not support filters", knnEngine));
        }

        String indexName = context.index().getName();
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
            .radius(engineRadius)
            .build();
        return KNNQueryFactory.create(createQueryRequest);
    }

    private ModelMetadata getModelMetadataForField(KNNVectorFieldMapper.KNNVectorFieldType knnVectorField) {
        String modelId = knnVectorField.getModelId();

        if (modelId == null) {
            throw new IllegalArgumentException(String.format("Field '%s' does not have model.", this.fieldName));
        }

        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (modelMetadata == null) {
            throw new IllegalArgumentException(String.format("Model ID '%s' does not exist.", modelId));
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

    /**
     * Builder for KNNQueryBuilder class to create KNNQueryBuilder, which can be used to build k-NN query or r-NN query.
     * */
    public static class Builder {
        private final String fieldName;
        private final float[] vector;
        private Integer k = null;
        private Float distance;
        private QueryBuilder filter;
        private boolean ignoreUnmapped;
        private String queryName;
        private float boost = AbstractQueryBuilder.DEFAULT_BOOST;

        public Builder(String fieldName, float[] vector) {
            if (Strings.isNullOrEmpty(fieldName)) {
                throw new IllegalArgumentException("[" + NAME + "] requires fieldName");
            }
            if (vector == null) {
                throw new IllegalArgumentException("[" + NAME + "] requires query vector");
            }
            if (vector.length == 0) {
                throw new IllegalArgumentException("[" + NAME + "] query vector is empty");
            }
            this.fieldName = fieldName;
            this.vector = vector;
        }

        public Builder k(Integer k) {
            this.k = k;
            return this;
        }

        public Builder distance(Float distance) {
            this.distance = distance;
            return this;
        }

        public Builder filter(QueryBuilder filter) {
            this.filter = filter;
            return this;
        }

        public Builder ignoreUnmapped(boolean ignoreUnmapped) {
            this.ignoreUnmapped = ignoreUnmapped;
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
            if ((k != null && distance != null) || (k == null && distance == null)) {
                throw new IllegalArgumentException("[" + NAME + "] requires either k or distance must be set");
            }
            if (k != null && (k <= 0 || k > K_MAX)) {
                throw new IllegalArgumentException("[" + NAME + "] requires 0 < k <= " + K_MAX);
            }
            if (distance != null && distance < 0) {
                throw new IllegalArgumentException("[" + NAME + "] requires distance >= 0");
            }

            return new KNNQueryBuilder(this);
        }
    }
}
