/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.search.Query;
import org.opensearch.common.ParseField;
import org.opensearch.common.ParsingException;
import org.opensearch.common.Strings;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryShardContext;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

/**
 * Helper class to build the KNN query
 */
public class KNNQueryBuilder extends AbstractQueryBuilder<KNNQueryBuilder> {
    private static Logger logger = LogManager.getLogger(KNNQueryBuilder.class);
    private static ModelDao modelDao;

    public static final ParseField VECTOR_FIELD = new ParseField("vector");
    public static final ParseField K_FIELD = new ParseField("k");
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

    /**
     * Constructs a new knn query
     *
     * @param fieldName Name of the filed
     * @param vector    Array of floating points
     * @param k         K nearest neighbours for the given vector
     */
    public KNNQueryBuilder(String fieldName, float[] vector, int k) {
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
        } catch (IOException ex) {
            throw new RuntimeException("[KNN] Unable to create KNNQueryBuilder: " + ex);
        }
    }

    public static KNNQueryBuilder fromXContent(XContentParser parser) throws IOException {
        String fieldName = null;
        List<Object> vector = null;
        float boost = AbstractQueryBuilder.DEFAULT_BOOST;
        int k = 0;
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

        KNNQueryBuilder knnQuery = new KNNQueryBuilder(fieldName, ObjectsToFloats(vector), k);
        knnQuery.queryName(queryName);
        knnQuery.boost(boost);
        return knnQuery;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(fieldName);
        out.writeFloatArray(vector);
        out.writeInt(k);
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

    @Override
    public void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.startObject(fieldName);

        builder.field(VECTOR_FIELD.getPreferredName(), vector);
        builder.field(K_FIELD.getPreferredName(), k);
        printBoostAndQueryName(builder);
        builder.endObject();
        builder.endObject();
    }

    @Override
    protected Query doToQuery(QueryShardContext context) throws IOException {

        MappedFieldType mappedFieldType = context.fieldMapper(this.fieldName);

        if (!(mappedFieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType)) {
            throw new IllegalArgumentException("Field '" + this.fieldName + "' is not knn_vector type.");
        }

        int dimension = ((KNNVectorFieldMapper.KNNVectorFieldType) mappedFieldType).getDimension();

        // If the dimension is not set, then the only valid route forward is if the field uses a model
        if (dimension == -1) {
            String modelId = ((KNNVectorFieldMapper.KNNVectorFieldType) mappedFieldType).getModelId();

            if (modelId == null) {
                throw new IllegalArgumentException("Field '" + this.fieldName + "' does not have dimension set.");
            }

            ModelMetadata modelMetadata = modelDao.getMetadata(modelId);

            if (modelMetadata == null) {
                throw new IllegalArgumentException("Model ID \"" + modelId + "\" does not exist.");
            }
            dimension = modelMetadata.getDimension();
        }

        if (dimension != vector.length) {
            throw new IllegalArgumentException(
                "Query vector has invalid dimension: " + vector.length + ". Dimension should be: " + dimension
            );
        }

        return new KNNQuery(this.fieldName, vector, k, context.index().getName());
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
