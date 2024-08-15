/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.opensearch.common.ValidationException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ObjectParser;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.util.IndexUtil;

import java.io.IOException;
import java.util.Locale;

import static org.opensearch.knn.index.query.KNNQueryBuilder.RESCORE_OVERSAMPLE_FIELD;

/**
 * Note: This parser is used by neural plugin as well, breaking changes will require changes in neural as well
 */
@Getter
@AllArgsConstructor
@Log4j2
public final class RescoreParser {

    public static final String RESCORE_PARAMETER = "rescore";
    public static final String RESCORE_OVERSAMPLE_PARAMETER = "oversample_factor";

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
     * Validate the rescore context
     *
     * @return ValidationException if validation fails, null otherwise
     */
    public static ValidationException validate(RescoreContext rescoreContext) {
        if (rescoreContext.getOversampleFactor() < RescoreContext.MIN_OVERSAMPLE_FACTOR) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Oversample factor [%f] cannot be less than [%f]",
                    rescoreContext.getOversampleFactor(),
                    RescoreContext.MIN_OVERSAMPLE_FACTOR
                )
            );
            return validationException;
        }

        if (rescoreContext.getOversampleFactor() > RescoreContext.MAX_OVERSAMPLE_FACTOR) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Oversample factor [%f] cannot be more than [%f]",
                    rescoreContext.getOversampleFactor(),
                    RescoreContext.MAX_OVERSAMPLE_FACTOR
                )
            );
            return validationException;
        }
        return null;
    }

    /**
     *
     * @param in stream input
     * @return RescoreContext
     * @throws IOException on stream failure
     */
    public static RescoreContext streamInput(StreamInput in) throws IOException {
        if (!IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), RESCORE_PARAMETER)) {
            return null;
        }
        Float oversample = in.readOptionalFloat();
        if (oversample == null) {
            return null;
        }
        return RescoreContext.builder().oversampleFactor(oversample).build();
    }

    /**
     *
     * @param out stream output
     * @param rescoreContext RescoreContext
     * @throws IOException on stream failure
     */
    public static void streamOutput(StreamOutput out, RescoreContext rescoreContext) throws IOException {
        if (!IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), RESCORE_PARAMETER)) {
            return;
        }
        out.writeOptionalFloat(rescoreContext == null ? null : rescoreContext.getOversampleFactor());
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
