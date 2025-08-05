/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.WithFieldName;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.parser.KNNExactQueryBuilderParser;
import org.opensearch.knn.index.util.IndexUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.common.KNNValidationUtil.validateByteVectorValue;

/**
 * Helper class to build the KNN exact query
 */
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@Log4j2
public class KNNExactQueryBuilder extends AbstractQueryBuilder<KNNExactQueryBuilder> implements WithFieldName {

    public static final ParseField VECTOR_FIELD = new ParseField("vector");
    public static final ParseField SPACE_TYPE_FIELD = new ParseField("space_type");
    public static final ParseField IGNORE_UNMAPPED_FIELD = new ParseField("ignore_unmapped");
    public static final ParseField EXPAND_NESTED_FIELD = new ParseField(EXPAND_NESTED);

    /**
     * The name for the knn exact query
     */
    public static final String NAME = "exact_knn";

    private final String fieldName;
    private final float[] vector;
    @Getter
    private String spaceType;
    @Getter
    private boolean ignoreUnmapped;
    @Getter
    private Boolean expandNested;

    public static class Builder {
        private String fieldName;
        private float[] vector;
        private String spaceType;
        private boolean ignoreUnmapped;
        private String queryName;
        private float boost = DEFAULT_BOOST;
        private Boolean expandNested;

        public Builder() {}

        public Builder fieldName(String fieldName) {
            this.fieldName = fieldName;
            return this;
        }

        public Builder vector(float[] vector) {
            this.vector = vector;
            return this;
        }

        public Builder spaceType(String spaceType) {
            this.spaceType = spaceType;
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

        public Builder expandNested(Boolean expandNested) {
            this.expandNested = expandNested;
            return this;
        }

        public KNNExactQueryBuilder build() {
            validate();
            return new KNNExactQueryBuilder(fieldName, vector, spaceType, ignoreUnmapped, expandNested).boost(boost).queryName(queryName);
        }

        private void validate() {
            KNNBuilderAndParserUtils.validateFieldName(fieldName, NAME);
            KNNBuilderAndParserUtils.validateVector(vector, NAME);

            if (spaceType != null) {
                try {
                    SpaceType.getSpace(spaceType);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "[%s] requires valid space type for exact search, refer to allowed space types.", NAME)
                    );
                }
            }
        }
    }

    public static KNNExactQueryBuilder.Builder builder() {
        return new KNNExactQueryBuilder.Builder();
    }

    /**
     * @param in Reads from stream
     * @throws IOException Throws IO Exception
     */
    public KNNExactQueryBuilder(StreamInput in) throws IOException {
        super(in);
        KNNExactQueryBuilder.Builder builder = KNNExactQueryBuilderParser.streamInput(in, IndexUtil::isClusterOnOrAfterMinRequiredVersion);
        fieldName = builder.fieldName;
        vector = builder.vector;
        spaceType = builder.spaceType;
        ignoreUnmapped = builder.ignoreUnmapped;
        expandNested = builder.expandNested;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        KNNExactQueryBuilderParser.streamOutput(out, this, IndexUtil::isClusterOnOrAfterMinRequiredVersion);
    }

    /**
     * @return The field name used in this query
     */
    @Override
    public String fieldName() {
        return this.fieldName;
    }

    /**
     * @return Returns the vector used in this query
     */
    public Object vector() {
        return this.vector;
    }

    @Override
    public void doXContent(XContentBuilder builder, Params params) throws IOException {
        KNNExactQueryBuilderParser.toXContent(builder, params, this);
    }

    @Override
    protected Query doToQuery(QueryShardContext context) {
        MappedFieldType mappedFieldType = KNNBuilderAndParserUtils.validateAndGetFieldType(this.fieldName, context, ignoreUnmapped);
        if (mappedFieldType == null) {
            return new MatchNoDocsQuery();
        }

        KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldType) mappedFieldType;
        KNNMappingConfig knnMappingConfig = knnVectorFieldType.getKnnMappingConfig();
        QueryConfigFromMapping queryConfigFromMapping = getQueryConfig(knnMappingConfig, knnVectorFieldType);

        SpaceType resolvedSpaceType = this.getSpaceType() != null
            ? SpaceType.getSpace(this.getSpaceType())
            : queryConfigFromMapping.getSpaceType();
        VectorDataType vectorDataType = queryConfigFromMapping.getVectorDataType();
        knnVectorFieldType.transformQueryVector(vector);

        final String indexName = context.index().getName();

        int vectorLength = VectorDataType.BINARY == vectorDataType ? vector.length * Byte.SIZE : vector.length;
        KNNBuilderAndParserUtils.validateVectorDimension(vectorLength, knnMappingConfig.getDimension());

        byte[] byteVector = new byte[0];
        switch (vectorDataType) {
            case BINARY:
                byteVector = new byte[vector.length];
                for (int i = 0; i < vector.length; i++) {
                    validateByteVectorValue(vector[i], knnVectorFieldType.getVectorDataType());
                    byteVector[i] = (byte) vector[i];
                }
                resolvedSpaceType.validateVector(byteVector);
                break;
            case BYTE:
                for (float v : vector) {
                    validateByteVectorValue(v, knnVectorFieldType.getVectorDataType());
                }
                resolvedSpaceType.validateVector(vector);
                break;
            default:
                resolvedSpaceType.validateVector(vector);
        }

        BitSetProducer parentFilter = context.getParentFilter();
        boolean isExpandNested = expandNested == null ? false : expandNested;
        KNNBuilderAndParserUtils.validateExpandNested(isExpandNested, parentFilter);

        log.debug("Creating exact k-NN query for index:{}, field:{}, spaceType:{}", indexName, fieldName, spaceType);

        KNNExactQuery knnExactQuery = null;
        switch (vectorDataType) {
            case BINARY:
                knnExactQuery = KNNExactQuery.builder()
                    .field(fieldName)
                    .byteQueryVector(byteVector)
                    .indexName(indexName)
                    .parentFilter(parentFilter)
                    .spaceType(resolvedSpaceType.getValue())
                    .vectorDataType(vectorDataType)
                    .expandNested(isExpandNested)
                    .build();
                break;
            default:
                knnExactQuery = KNNExactQuery.builder()
                    .field(fieldName)
                    .queryVector(vector)
                    .byteQueryVector(byteVector)
                    .indexName(indexName)
                    .parentFilter(parentFilter)
                    .spaceType(resolvedSpaceType.getValue())
                    .vectorDataType(vectorDataType)
                    .expandNested(isExpandNested)
                    .build();
        }
        return knnExactQuery;
    }

    private KNNExactQueryBuilder.QueryConfigFromMapping getQueryConfig(
        final KNNMappingConfig knnMappingConfig,
        final KNNVectorFieldType knnVectorFieldType
    ) {
        if (knnMappingConfig.getKnnMethodContext().isPresent()) {
            KNNMethodContext knnMethodContext = knnMappingConfig.getKnnMethodContext().get();
            return new KNNExactQueryBuilder.QueryConfigFromMapping(knnMethodContext.getSpaceType(), knnVectorFieldType.getVectorDataType());
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' is not built for exact search.", this.fieldName));
    }

    @Override
    protected boolean doEquals(KNNExactQueryBuilder other) {
        return Objects.equals(fieldName, other.fieldName)
            && Arrays.equals(vector, other.vector)
            && Objects.equals(ignoreUnmapped, other.ignoreUnmapped)
            && Objects.equals(expandNested, other.expandNested);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, Arrays.hashCode(vector), ignoreUnmapped, expandNested);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Getter
    @AllArgsConstructor
    private static class QueryConfigFromMapping {
        private final SpaceType spaceType;
        private final VectorDataType vectorDataType;
    }
}
