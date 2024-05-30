/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.query.parser;

import com.google.common.collect.ImmutableSet;
import lombok.Getter;
import lombok.SneakyThrows;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.model.AlgoQueryParameters;
import org.opensearch.knn.index.query.model.HNSWAlgoQueryParameters;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.index.IndexUtil.isClusterOnOrAfterMinRequiredVersion;
import static org.opensearch.knn.index.query.KNNQueryBuilder.EF_SEARCH_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.NAME;
import static org.opensearch.knn.index.query.parser.KNNXParserUtil.parseJsonObject;

@Getter
public class MethodParametersParser {

    private static final Set<String> VALID_METHOD_PARAMETERS = ImmutableSet.of(METHOD_PARAMETER_EF_SEARCH);

    @SneakyThrows(IOException.class)
    public static AlgoQueryParameters streamInMethodParameters(StreamInput in) {
        Integer efSearch = null;
        if (isClusterOnOrAfterMinRequiredVersion(METHOD_PARAMETER_EF_SEARCH)) {
            efSearch = in.readOptionalInt();
        }
        return efSearch != null ? HNSWAlgoQueryParameters.builder().efSearch(efSearch).build() : null;
    }

    @SneakyThrows(IOException.class)
    public static void streamOutMethodParameters(final StreamOutput out, final AlgoQueryParameters algoQueryParameters) {
        final Optional<HNSWAlgoQueryParameters> hnswMethodParameters = extractHnswAlgoParameters(algoQueryParameters);

        if (isClusterOnOrAfterMinRequiredVersion(METHOD_PARAMETER_EF_SEARCH)) {
            // Write false even if it doesn't cast to hnsw, so we can deserialize w/o ambiguity as parameters get added
            out.writeOptionalInt(hnswMethodParameters.flatMap(HNSWAlgoQueryParameters::getEfSearch).orElse(null));
        }
    }

    public static Optional<HNSWAlgoQueryParameters> extractHnswAlgoParameters(final AlgoQueryParameters algoQueryParameters) {
        return Optional.ofNullable(algoQueryParameters)
            .filter(HNSWAlgoQueryParameters.class::isInstance)
            .map(HNSWAlgoQueryParameters.class::cast);
    }

    @SneakyThrows(IOException.class)
    public static AlgoQueryParameters fromXContent(final XContentParser parser) {
        final Map<String, Object> parameters = parseJsonObject(parser);
        if (parameters.isEmpty()) {
            throw new IllegalArgumentException("[" + NAME + "] method_parameter cannot be empty");
        }

        for (String jsonkey : parameters.keySet()) {
            if (!VALID_METHOD_PARAMETERS.contains(jsonkey)) {
                throw new IllegalArgumentException("[" + NAME + "] unknown parameter " + jsonkey + " found.");
            }
        }

        return Optional.ofNullable((Integer) parameters.get(METHOD_PARAMETER_EF_SEARCH))
            .filter(ef -> EF_SEARCH_FIELD.match(METHOD_PARAMETER_EF_SEARCH, parser.getDeprecationHandler()))
            .map(ef -> HNSWAlgoQueryParameters.builder().efSearch(ef).build())
            .orElse(null);
    }

    @SneakyThrows(IOException.class)
    public static XContentBuilder toXContent(final XContentBuilder xContentBuilder, final AlgoQueryParameters algoQueryParameters) {

        final HNSWAlgoQueryParameters hnswAlgoParameters = Optional.ofNullable(algoQueryParameters)
            .filter(HNSWAlgoQueryParameters.class::isInstance)
            .map(HNSWAlgoQueryParameters.class::cast)
            .orElse(null);

        if (hnswAlgoParameters != null && hnswAlgoParameters.getEfSearch().isPresent()) {
            xContentBuilder.startObject(METHOD_PARAMETER);
            xContentBuilder.field(EF_SEARCH_FIELD.getPreferredName(), hnswAlgoParameters.getEfSearch().get());
            xContentBuilder.endObject();
        }
        return xContentBuilder;
    }
}
