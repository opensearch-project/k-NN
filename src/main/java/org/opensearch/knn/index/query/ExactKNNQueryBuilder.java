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
import org.opensearch.knn.index.query.parser.ExactKNNQueryBuilderParser;
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
public class ExactKNNQueryBuilder extends AbstractQueryBuilder<ExactKNNQueryBuilder> implements WithFieldName {

    public static final ParseField VECTOR_FIELD = new ParseField("vector");
    public static final ParseField SPACE_TYPE_FIELD = new ParseField("space_type");
    public static final ParseField IGNORE_UNMAPPED_FIELD = new ParseField("ignore_unmapped");
    public static final ParseField EXPAND_NESTED_FIELD = new ParseField(EXPAND_NESTED);

    /**
     * The name for the knn exact query
     */
    public static final String NAME = "exact_knn";

    private final String fieldName;
    @Getter
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

        public ExactKNNQueryBuilder build() {
            validate();
            return new ExactKNNQueryBuilder(fieldName, vector, spaceType, ignoreUnmapped, expandNested).boost(boost).queryName(queryName);
        }

        private void validate() {
            KNNBuilderUtils.validateFieldName(fieldName, NAME);
            KNNBuilderUtils.validateVector(vector, NAME);

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

    public static ExactKNNQueryBuilder.Builder builder() {
        return new ExactKNNQueryBuilder.Builder();
    }

    /**
     * @param in Reads from stream
     * @throws IOException Throws IO Exception
     */
    public ExactKNNQueryBuilder(StreamInput in) throws IOException {
        super(in);
        ExactKNNQueryBuilder.Builder builder = ExactKNNQueryBuilderParser.streamInput(in, IndexUtil::isClusterOnOrAfterMinRequiredVersion);
        fieldName = builder.fieldName;
        vector = builder.vector;
        spaceType = builder.spaceType;
        ignoreUnmapped = builder.ignoreUnmapped;
        expandNested = builder.expandNested;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        ExactKNNQueryBuilderParser.streamOutput(out, this, IndexUtil::isClusterOnOrAfterMinRequiredVersion);
    }

    /**
     * @return The field name used in this query
     */
    @Override
    public String fieldName() {
        return this.fieldName;
    }

    @Override
    public void doXContent(XContentBuilder builder, Params params) throws IOException {
        ExactKNNQueryBuilderParser.toXContent(builder, params, this);
    }

    @Override
    protected Query doToQuery(QueryShardContext context) {
        MappedFieldType mappedFieldType = KNNBuilderUtils.validateAndGetFieldType(this.fieldName, context, ignoreUnmapped);
        if (mappedFieldType == null) {
            log.debug("No mappedFieldType found for field [{}]", this.fieldName);
            return new MatchNoDocsQuery();
        }

        KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldType) mappedFieldType;
        KNNMappingConfig knnMappingConfig = knnVectorFieldType.getKnnMappingConfig();
        KNNMethodContext knnMethodContext;
        if (knnMappingConfig.getKnnMethodContext().isPresent()) {
            knnMethodContext = knnMappingConfig.getKnnMethodContext().get();
        } else {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' is not built for exact search.", this.fieldName));
        }

        SpaceType resolvedSpaceType = this.getSpaceType() != null
            ? SpaceType.getSpace(this.getSpaceType())
            : knnMethodContext.getSpaceType();
        VectorDataType vectorDataType = knnVectorFieldType.getVectorDataType();
        knnVectorFieldType.transformQueryVector(vector);

        final String indexName = context.index().getName();

        KNNBuilderUtils.validateVectorDimension(vectorDataType, vector.length, knnMappingConfig.getDimension());

        BitSetProducer parentFilter = context.getParentFilter();
        log.debug("Creating exact k-NN query for index:{}, field:{}, spaceType:{}", indexName, fieldName, spaceType);

        switch (vectorDataType) {
            case BINARY:
                byte[] byteVector = new byte[vector.length];
                for (int i = 0; i < vector.length; i++) {
                    validateByteVectorValue(vector[i], knnVectorFieldType.getVectorDataType());
                    byteVector[i] = (byte) vector[i];
                }
                // validate byteVector here because binary/hamming does not support float vectors
                resolvedSpaceType.validateVector(byteVector);
                return new ExactKNNByteQuery(
                    fieldName,
                    resolvedSpaceType.getValue(),
                    indexName,
                    vectorDataType,
                    parentFilter,
                    expandNested == null ? false : expandNested,
                    byteVector
                );
            // FloatQuery used for bytes + floats because bytes are packed in floats
            case BYTE, FLOAT:
                resolvedSpaceType.validateVector(vector);
                return new ExactKNNFloatQuery(
                    fieldName,
                    resolvedSpaceType.getValue(),
                    indexName,
                    vectorDataType,
                    parentFilter,
                    expandNested == null ? false : expandNested,
                    vector
                );
            default:
                throw new IllegalStateException("Unsupported vector data type found.");
        }
    }

    @Override
    protected boolean doEquals(ExactKNNQueryBuilder other) {
        return Objects.equals(fieldName, other.fieldName)
            && Arrays.equals(vector, other.vector)
            && Objects.equals(ignoreUnmapped, other.ignoreUnmapped)
            && Objects.equals(spaceType, other.spaceType)
            && Objects.equals(expandNested, other.expandNested);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, Arrays.hashCode(vector), ignoreUnmapped, spaceType, expandNested);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
