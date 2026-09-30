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
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.util.IndexUtil;

import java.io.IOException;
import java.util.Locale;

/**
 * Note: This parser is used by neural plugin as well, breaking changes will require changes in neural as well
 */
@Getter
@AllArgsConstructor
@Log4j2
public final class RescoreParser {

    public static final String RESCORE_PARAMETER = "rescore";
    public static final String RESCORE_OVERSAMPLE_PARAMETER = "oversample_factor";
    public static final String RESCORE_ENABLED_PARAMETER = "rescore_enabled";
    // Late-interaction (multi-vector MaxSim) rescore. Per k-NN RFC #3439, the target field is a DYNAMIC
    // key inside the rescore object (the late_interaction field name), whose value is { vector, similarity }.
    public static final String LATE_INTERACTION_VECTOR_PARAMETER = "vector";
    public static final String LATE_INTERACTION_SIMILARITY_PARAMETER = "similarity";
    // Default oversample factor for a late-interaction rescore (RFC #3439 Q4 interim; matches disk-mode).
    public static final float LATE_INTERACTION_DEFAULT_OVERSAMPLE_FACTOR = 3.0f;
    // Version-gate key for the late-interaction rescore wire payload (registered in IndexUtil).
    public static final String LATE_INTERACTION_RESCORE_FEATURE = "late_interaction_rescore";

    /**
     * Parses the {@code rescore} object. Recognizes {@code oversample_factor} (float); any OTHER object
     * key is treated as a late-interaction target field name (RFC #3439 shape):
     * <pre>
     * "rescore": {
     *   "oversample_factor": 3,
     *   "&lt;li_field&gt;": { "vector": [[..],[..], ...], "similarity": "maxSimDotProduct" }
     * }
     * </pre>
     */
    public static RescoreContext fromXContent(final XContentParser parser) {
        final RescoreContext.RescoreContextBuilder builder = RescoreContext.builder();
        boolean lateInteraction = false;
        try {
            XContentParser.Token token = parser.currentToken();
            // Ensure we are positioned ON the opening START_OBJECT, then step inside it.
            if (token == null) {
                token = parser.nextToken();
            }
            if (token == XContentParser.Token.START_OBJECT) {
                token = parser.nextToken();
            }
            String currentField = null;
            for (; token != null && token != XContentParser.Token.END_OBJECT; token = parser.nextToken()) {
                if (token == XContentParser.Token.FIELD_NAME) {
                    currentField = parser.currentName();
                } else if (RESCORE_OVERSAMPLE_PARAMETER.equals(currentField)) {
                    builder.oversampleFactor(parser.floatValue());
                } else if (token == XContentParser.Token.START_OBJECT) {
                    // Dynamic key = the late-interaction field name being rescored.
                    if (lateInteraction) {
                        throw new IllegalArgumentException("Only one late-interaction rescore field is supported");
                    }
                    parseLateInteractionField(parser, currentField, builder);
                    lateInteraction = true;
                } else {
                    throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "Unknown [%s] parameter [%s]", RESCORE_PARAMETER, currentField)
                    );
                }
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to parse [" + RESCORE_PARAMETER + "]", e);
        }
        // A late-interaction rescore defaults to a higher oversample factor than disk-mode rescore.
        if (lateInteraction && builder.build().getOversampleFactor() == RescoreContext.DEFAULT_OVERSAMPLE_FACTOR) {
            builder.oversampleFactor(LATE_INTERACTION_DEFAULT_OVERSAMPLE_FACTOR).userProvided(false);
        }
        return builder.build();
    }

    /** Parses a {@code "<li_field>": { "vector": [[..]], "similarity": "..." }} entry onto the builder. */
    private static void parseLateInteractionField(
        final XContentParser parser,
        final String fieldName,
        final RescoreContext.RescoreContextBuilder builder
    ) throws IOException {
        if (fieldName == null) {
            throw new IllegalArgumentException("Late-interaction rescore field name cannot be null");
        }
        float[][] vectors = null;
        String similarity = null;
        XContentParser.Token token;
        String currentField = null;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentField = parser.currentName();
            } else if (LATE_INTERACTION_VECTOR_PARAMETER.equals(currentField)) {
                vectors = parseMultiVector(parser);
            } else if (LATE_INTERACTION_SIMILARITY_PARAMETER.equals(currentField)) {
                similarity = parser.text();
            } else {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Unknown field [%s] in late-interaction rescore [%s]", currentField, fieldName)
                );
            }
        }
        if (vectors == null || vectors.length == 0) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] is required and must be non-empty in late-interaction rescore [%s]",
                    LATE_INTERACTION_VECTOR_PARAMETER,
                    fieldName
                )
            );
        }
        builder.lateInteractionField(fieldName);
        builder.lateInteractionQueryVectors(vectors);
        builder.lateInteractionSimilarity(similarity);
    }

    /** Parses a {@code [[..],[..], ...]} multi-vector into a {@code float[][]}. */
    private static float[][] parseMultiVector(final XContentParser parser) throws IOException {
        if (parser.currentToken() != XContentParser.Token.START_ARRAY) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "[%s] must be an array of vectors", LATE_INTERACTION_VECTOR_PARAMETER)
            );
        }
        final java.util.List<float[]> vectors = new java.util.ArrayList<>();
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_ARRAY) {
            if (token != XContentParser.Token.START_ARRAY) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] must be a list of numeric vectors", LATE_INTERACTION_VECTOR_PARAMETER)
                );
            }
            final java.util.List<Float> vec = new java.util.ArrayList<>();
            while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                vec.add(parser.floatValue());
            }
            final float[] arr = new float[vec.size()];
            for (int i = 0; i < vec.size(); i++) {
                arr[i] = vec.get(i);
            }
            vectors.add(arr);
        }
        return vectors.toArray(new float[0][]);
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
        final RescoreContext.RescoreContextBuilder builder = RescoreContext.builder().oversampleFactor(oversample);
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), RESCORE_ENABLED_PARAMETER)) {
            builder.rescoreEnabled(in.readBoolean());
        }
        // Late-interaction payload (optional), only present on the wire when the peer supports it.
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), LATE_INTERACTION_RESCORE_FEATURE)) {
            final String liField = in.readOptionalString();
            if (liField != null) {
                builder.lateInteractionField(liField);
                final int numTokens = in.readVInt();
                final float[][] queryVectors = new float[numTokens][];
                for (int i = 0; i < numTokens; i++) {
                    queryVectors[i] = in.readFloatArray();
                }
                builder.lateInteractionQueryVectors(queryVectors);
                builder.lateInteractionSimilarity(in.readOptionalString());
            }
        }
        return builder.build();
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
        if (rescoreContext != null && IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), RESCORE_ENABLED_PARAMETER)) {
            out.writeBoolean(rescoreContext.isRescoreEnabled());
        }
        // The reader returns early when the oversample factor is null, so the late-interaction block is
        // only on the wire when there is a context AND the peer version supports the feature. This keeps
        // the read/write framing symmetric with streamInput.
        if (rescoreContext == null) {
            return;
        }
        if (!IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), LATE_INTERACTION_RESCORE_FEATURE)) {
            return;
        }
        out.writeOptionalString(rescoreContext.getLateInteractionField());
        if (rescoreContext.getLateInteractionField() != null) {
            final float[][] queryVectors = rescoreContext.getLateInteractionQueryVectors();
            out.writeVInt(queryVectors.length);
            for (float[] token : queryVectors) {
                out.writeFloatArray(token);
            }
            out.writeOptionalString(rescoreContext.getLateInteractionSimilarity());
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
        if (rescoreContext.isLateInteraction()) {
            // Dynamic key = the late-interaction field name (RFC #3439 shape).
            builder.startObject(rescoreContext.getLateInteractionField());
            builder.startArray(LATE_INTERACTION_VECTOR_PARAMETER);
            for (float[] token : rescoreContext.getLateInteractionQueryVectors()) {
                builder.startArray();
                for (float v : token) {
                    builder.value(v);
                }
                builder.endArray();
            }
            builder.endArray();
            if (rescoreContext.getLateInteractionSimilarity() != null) {
                builder.field(LATE_INTERACTION_SIMILARITY_PARAMETER, rescoreContext.getLateInteractionSimilarity());
            }
            builder.endObject();
        }
        builder.endObject();
    }
}
