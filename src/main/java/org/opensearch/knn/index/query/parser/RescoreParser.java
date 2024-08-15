/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ObjectParser;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.RESCORE_OVERSAMPLE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.RESCORE_PARAMETER;
import static org.opensearch.knn.index.query.KNNQueryBuilder.RESCORE_OVERSAMPLE_FIELD;

/**
 * Note: This parser is used by neural plugin as well, breaking changes will require changes in neural as well
 */
@Getter
@AllArgsConstructor
@Log4j2
public final class RescoreParser {

    private static final ObjectParser<RescoreContext.RescoreContextBuilder, Void> INTERNAL_PARSER = createInternalObjectParser();

    private static ObjectParser<RescoreContext.RescoreContextBuilder, Void> createInternalObjectParser() {
        ObjectParser<RescoreContext.RescoreContextBuilder, Void> internalParser = new ObjectParser<>(
            RESCORE_PARAMETER,
            RescoreContext::builder
        );
        internalParser.declareFloat(RescoreContext.RescoreContextBuilder::oversampleFactor, RESCORE_OVERSAMPLE_FIELD);
        return internalParser;
    }

    /**
     *
     * @param in stream input
     * @param minClusterVersionCheck function to check if the cluster version meets the minimum requirement
     * @return RescoreContext
     * @throws IOException on stream failure
     */
    public static RescoreContext streamInput(StreamInput in, Function<String, Boolean> minClusterVersionCheck) throws IOException {
        if (!in.readBoolean()) {
            return null;
        }

        RescoreContext.RescoreContextBuilder builder = RescoreContext.builder();
        if (minClusterVersionCheck.apply(RESCORE_PARAMETER)) {
            builder.oversampleFactor(in.readFloat());
        }
        return builder.build();
    }

    /**
     *
     * @param out stream output
     * @param rescoreContext RescoreContext
     * @param minClusterVersionCheck function to check if the cluster version meets the minimum requirement
     * @throws IOException on stream failure
     */
    public static void streamOutput(StreamOutput out, RescoreContext rescoreContext, Function<String, Boolean> minClusterVersionCheck)
        throws IOException {
        if (rescoreContext == null) {
            out.writeBoolean(false);
            return;
        }

        out.writeBoolean(true);
        if (minClusterVersionCheck.apply(RESCORE_PARAMETER)) {
            out.writeFloat(rescoreContext.getOversampleFactor());
        }
    }

    /**
     *
     * @param builder XContentBuilder
     * @param rescoreContext RescoreContext
     * @throws IOException on XContent failure
     */
    public static void doXContent(final XContentBuilder builder, final RescoreContext rescoreContext) throws IOException {
        builder.startObject(RESCORE_PARAMETER);
        builder.field(RESCORE_OVERSAMPLE_PARAMETER, rescoreContext.getOversampleFactor());
        builder.endObject();
    }

    /**
     *
     * @param parser input parser
     * @return RescoreContext
     */
    public static RescoreContext fromXContent(final XContentParser parser) {
        return INTERNAL_PARSER.apply(parser, null).build();
    }
}
