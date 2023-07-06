/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.opensearch.core.common.Strings;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.apache.lucene.search.Query;
import org.opensearch.core.ParseField;
import org.opensearch.common.ParsingException;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryShardContext;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

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
    public static int K_MAX = 10000;
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
    private QueryBuilder filter;

    /**
     * Constructs a new knn query
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
    }

    public static void initialize(ModelDao modelDao) {
        KNNQueryBuilder.modelDao = modelDao;
    }

    private static float[] ObjectsToFloats(List<Object> objs) {
        float[] vec = new float[objs.size()];
        for (int i = 0; i < objs.size(); i++) {
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
        } catch (IOException ex) {
            throw new RuntimeException("[KNN] Unable to create KNNQueryBuilder", ex);
        }
    }

    public static KNNQueryBuilder fromXContent(XContentParser parser) throws IOException {
        String fieldName = null;
        List<Object> vector = null;
        float boost = AbstractQueryBuilder.DEFAULT_BOOST;
        int k = 0;
        QueryBuilder filter = null;
        String queryName = null;
        String currentFieldName = null;
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
                        } else if (AbstractQueryBuilder.NAME_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            queryName = parser.text();
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

        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(fieldName, ObjectsToFloats(vector), k, filter);
        knnQueryBuilder.queryName(queryName);
        knnQueryBuilder.boost(boost);
        return knnQueryBuilder;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(fieldName);
        out.writeFloatArray(vector);
        out.writeInt(k);
        out.writeOptionalNamedWriteable(filter);
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

    public QueryBuilder getFilter() {
        return this.filter;
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
        printBoostAndQueryName(builder);
        builder.endObject();
        builder.endObject();
    }

    @Override
    protected Query doToQuery(QueryShardContext context) {
        MappedFieldType mappedFieldType = context.fieldMapper(this.fieldName);

        if (!(mappedFieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType)) {
            throw new IllegalArgumentException(String.format("Field '%s' is not knn_vector type.", this.fieldName));
        }

        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldMapper.KNNVectorFieldType) mappedFieldType;
        int fieldDimension = knnVectorFieldType.getDimension();
        KNNMethodContext knnMethodContext = knnVectorFieldType.getKnnMethodContext();
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        VectorDataType vectorDataType = knnVectorFieldType.getVectorDataType();

        if (fieldDimension == -1) {
            // If dimension is not set, the field uses a model and the information needs to be retrieved from there
            ModelMetadata modelMetadata = getModelMetadataForField(knnVectorFieldType);
            fieldDimension = modelMetadata.getDimension();
            knnEngine = modelMetadata.getKnnEngine();
        } else if (knnMethodContext != null) {
            // If the dimension is set but the knnMethodContext is not then the field is using the legacy mapping
            knnEngine = knnMethodContext.getKnnEngine();
        }

        if (fieldDimension != vector.length) {
            throw new IllegalArgumentException(
                String.format("Query vector has invalid dimension: %d. Dimension should be: %d", vector.length, fieldDimension)
            );
        }

        byte[] byteVector = new byte[0];
        if (VectorDataType.BYTE.equals(vectorDataType)) {
            byteVector = new byte[vector.length];
            for (int i = 0; i < vector.length; i++) {
                validateByteVectorValue(vector[i]);
                byteVector[i] = (byte) vector[i];
            }
        }

        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine) && filter != null && knnEngine != KNNEngine.FAISS) {
            throw new IllegalArgumentException(String.format("Engine [%s] does not support filters", knnEngine));
        }

        String indexName = context.index().getName();
        KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
            .knnEngine(knnEngine)
            .indexName(indexName)
            .fieldName(this.fieldName)
            .vector(VectorDataType.FLOAT.equals(vectorDataType) ? this.vector : null)
            .byteVector(VectorDataType.BYTE.equals(vectorDataType) ? byteVector : null)
            .vectorDataType(vectorDataType)
            .k(this.k)
            .filter(this.filter)
            .context(context)
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
        return Objects.equals(fieldName, other.fieldName) && Objects.equals(vector, other.vector) && Objects.equals(k, other.k);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, vector, k);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
