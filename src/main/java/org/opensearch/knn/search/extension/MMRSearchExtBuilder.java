/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.extension;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.opensearch.knn.search.processor.mmr.MMROverSampleProcessor;
import org.opensearch.knn.search.processor.mmr.MMRRerankProcessor;
import org.opensearch.search.SearchExtBuilder;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.CANDIDATES;
import static org.opensearch.knn.common.KNNConstants.DIVERSITY;
import static org.opensearch.knn.common.KNNConstants.MMR;
import static org.opensearch.knn.common.KNNConstants.VECTOR_FIELD_DATA_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_FIELD_PATH;
import static org.opensearch.knn.common.KNNConstants.VECTOR_FIELD_SPACE_TYPE;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;
import static org.opensearch.search.pipeline.SearchPipelineService.ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING;

/**
 * Search extension for Maximal Marginal Relevance
 */
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@Getter
public class MMRSearchExtBuilder extends SearchExtBuilder {

    public static final String NAME = MMR;

    // Used to control the weight of the diversity, range is from [0,1]. (diversity = 1) prioritizes maximum diversity
    // which means the documents are selected just based on how different they are from already chosen documents.
    public static final ParseField DIVERSITY_FIELD = new ParseField(DIVERSITY);
    // Used to control how many candidates we should oversample for MMR
    public static final ParseField CANDIDATES_FIELD = new ParseField(CANDIDATES);
    // Path to the vector field used for MMR re-rank. Optional. If not provided we should auto resolve it based on the
    // search request.
    public static final ParseField VECTOR_FIELD_PATH_FIELD = new ParseField(VECTOR_FIELD_PATH);
    // Data type of the vector field. Used to decide how to parse the vector field to calculate the similarity.
    // Optional. If not provided we should auto resolve it from the index mapping.
    public static final ParseField VECTOR_FIELD_DATA_TYPE_FIELD = new ParseField(VECTOR_FIELD_DATA_TYPE);
    // Space type of the vector field which is used to decide the similarity function. Optional. If not provided we
    // should auto resolve it from the index mapping.
    public static final ParseField VECTOR_FIELD_SPACE_TYPE_FIELD = new ParseField(VECTOR_FIELD_SPACE_TYPE);

    private Float diversity;
    private Integer candidates;
    private String vectorFieldPath;
    private VectorDataType vectorFieldDataType;
    private SpaceType spaceType;

    public static class Builder {
        private Float diversity;
        private Integer candidates;
        private String vectorFieldPath;
        private VectorDataType vectorFieldDataType;
        private SpaceType spaceType;

        public Builder() {}

        public Builder diversity(Float diversity) {
            this.diversity = diversity;
            return this;
        }

        public Builder candidates(Integer candidates) {
            this.candidates = candidates;
            return this;
        }

        public Builder vectorFieldPath(String vectorFieldPath) {
            this.vectorFieldPath = vectorFieldPath;
            return this;
        }

        public Builder vectorFieldDataType(String vectorFieldDataType) {
            try {
                this.vectorFieldDataType = VectorDataType.valueOf(vectorFieldDataType.toUpperCase(Locale.ROOT));
                return this;
            } catch (Exception e) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "%s in mmr query extension is not valid, valid values are %s.",
                        VECTOR_FIELD_DATA_TYPE_FIELD.getPreferredName(),
                        SUPPORTED_VECTOR_DATA_TYPES
                    )
                );
            }
        }

        public Builder spaceType(String spaceType) {
            if (!Arrays.stream(SpaceType.VALID_VALUES).toList().contains(spaceType)) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "%s in mmr query extension is not valid, valid values are %s.",
                        VECTOR_FIELD_SPACE_TYPE_FIELD.getPreferredName(),
                        String.join(",", SpaceType.VALID_VALUES)
                    )
                );
            }
            this.spaceType = SpaceType.getSpace(spaceType);
            return this;
        }

        public MMRSearchExtBuilder build() {
            setDefault();
            validate();
            return new MMRSearchExtBuilder(diversity, candidates, vectorFieldPath, vectorFieldDataType, spaceType);
        }

        private void setDefault() {
            if (diversity == null) {
                diversity = 0.5f;
            }
        }

        private void validate() {
            if (diversity < 0.0 || diversity > 1.0) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "%s in mmr query extension must be between 0.0 and 1.0", DIVERSITY_FIELD.getPreferredName())
                );
            }

            if (candidates != null && candidates < 0) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "%s in mmr query extension must be larger than 0.", CANDIDATES_FIELD.getPreferredName())
                );
            }

            if (vectorFieldPath != null && vectorFieldPath.isEmpty()) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "%s in mmr query extension should not be an empty string.",
                        VECTOR_FIELD_PATH_FIELD.getPreferredName()
                    )
                );
            }
        }
    }

    public MMRSearchExtBuilder(StreamInput in) throws IOException {
        diversity = in.readOptionalFloat();
        candidates = in.readOptionalVInt();
        vectorFieldPath = in.readOptionalString();
        String vectorFieldDataTypeStr = in.readOptionalString();
        if (vectorFieldDataTypeStr != null) {
            vectorFieldDataType = VectorDataType.get(vectorFieldDataTypeStr);
        }
        String spaceTypeStr = in.readOptionalString();
        if (spaceTypeStr != null) {
            spaceType = SpaceType.getSpace(spaceTypeStr);
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeOptionalFloat(diversity);
        out.writeOptionalVInt(candidates);
        out.writeOptionalString(vectorFieldPath);
        out.writeOptionalString(vectorFieldDataType == null ? null : vectorFieldDataType.getValue());
        out.writeOptionalString(spaceType == null ? null : spaceType.getValue());
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        if (diversity != null) {
            builder.field(DIVERSITY_FIELD.getPreferredName(), diversity);
        }
        if (candidates != null) {
            builder.field(CANDIDATES_FIELD.getPreferredName(), candidates);
        }
        if (vectorFieldPath != null) {
            builder.field(VECTOR_FIELD_PATH_FIELD.getPreferredName(), vectorFieldPath);
        }
        if (vectorFieldDataType != null) {
            builder.field(VECTOR_FIELD_DATA_TYPE_FIELD.getPreferredName(), vectorFieldDataType.getValue());
        }
        if (spaceType != null) {
            builder.field(VECTOR_FIELD_SPACE_TYPE_FIELD.getPreferredName(), spaceType.getValue());
        }
        builder.endObject();
        return builder;
    }

    @Override
    public int hashCode() {
        return Objects.hash(diversity, candidates, vectorFieldPath, vectorFieldDataType, spaceType);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        EqualsBuilder equalsBuilder = new EqualsBuilder();
        MMRSearchExtBuilder other = (MMRSearchExtBuilder) obj;
        equalsBuilder.append(diversity, other.diversity);
        equalsBuilder.append(candidates, other.candidates);
        equalsBuilder.append(vectorFieldPath, other.vectorFieldPath);
        equalsBuilder.append(vectorFieldDataType, other.vectorFieldDataType);
        equalsBuilder.append(spaceType, other.spaceType);
        return equalsBuilder.isEquals();
    }

    public static MMRSearchExtBuilder parse(XContentParser parser) throws IOException {
        ensureMMRProcessorsEnabled();
        XContentParser.Token token;
        String currentFieldName = "";
        Builder builder = new Builder();
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token.isValue()) {
                if (DIVERSITY_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.diversity(parser.floatValue());
                } else if (CANDIDATES_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.candidates(parser.intValue());
                } else if (VECTOR_FIELD_PATH_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.vectorFieldPath(parser.text());
                } else if (VECTOR_FIELD_DATA_TYPE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.vectorFieldDataType(parser.text());
                } else if (VECTOR_FIELD_SPACE_TYPE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.spaceType(parser.text());
                } else {
                    throw unsupportedField(parser, currentFieldName);
                }
            } else {
                throw unsupportedField(parser, currentFieldName);
            }
        }
        return builder.build();
    }

    private static void ensureMMRProcessorsEnabled() {
        List<String> enabledFactories = KNNClusterUtil.instance().getEnabledSystemGeneratedFactories();
        boolean isMMRProcessorsEnabled = enabledFactories.contains("*")
            || (enabledFactories.contains(MMROverSampleProcessor.MMROverSampleProcessorFactory.TYPE)
                && enabledFactories.contains(MMRRerankProcessor.MMRRerankProcessorFactory.TYPE));
        if (isMMRProcessorsEnabled == false) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "We need to enable [%s, %s] in the" + " cluster setting [%s] to support the mmr search extension.",
                    MMROverSampleProcessor.MMROverSampleProcessorFactory.TYPE,
                    MMRRerankProcessor.MMRRerankProcessorFactory.TYPE,
                    ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING.getKey()
                )
            );
        }
    }

    private static ParsingException unsupportedField(XContentParser parser, String fieldName) {
        return new ParsingException(
            parser.getTokenLocation(),
            String.format(Locale.ROOT, "[%s] query extension does not support [%s]", NAME, fieldName)
        );
    }

}
