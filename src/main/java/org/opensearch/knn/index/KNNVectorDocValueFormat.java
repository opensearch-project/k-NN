/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.Getter;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.search.DocValueFormat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * DocValueFormat for knn_vector fields. Supports two modes:
 * <ul>
 *   <li>{@code array} — vectors are returned as JSON numeric arrays</li>
 *   <li>{@code binary} (default) — vectors are returned as byte arrays that XContentBuilder
 *       base64-encodes during serialization</li>
 * </ul>
 */
@Getter
public enum KNNVectorDocValueFormat implements DocValueFormat {

    ARRAY_FORMAT("array", false),
    BINARY_FORMAT("binary", true);

    public static final String NAME = "knn_vector";

    private final String formatName;
    private final boolean binary;

    KNNVectorDocValueFormat(final String formatName, boolean binary) {
        this.formatName = formatName;
        this.binary = binary;
    }

    public static KNNVectorDocValueFormat fromStream(final StreamInput in) throws IOException {
        return in.readBoolean() ? BINARY_FORMAT : ARRAY_FORMAT;
    }

    /**
     * Resolves the format string to the corresponding enum constant.
     * Returns {@link #BINARY_FORMAT} when format is null (the default).
     *
     * @param format the format string from the docvalue_fields request, or null for default
     * @return the matching {@link KNNVectorDocValueFormat}
     * @throws IllegalArgumentException if the format string is not recognized
     */
    public static KNNVectorDocValueFormat fromFormatString(final String format) {
        if (format == null || BINARY_FORMAT.formatName.equals(format)) {
            return BINARY_FORMAT;
        }
        if (ARRAY_FORMAT.formatName.equals(format)) {
            return ARRAY_FORMAT;
        }
        throw new IllegalArgumentException(
            "Unsupported knn_vector docvalue_fields format ["
                + format
                + "]. Supported formats are "
                + Arrays.toString(KNNVectorDocValueFormat.values())
        );
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public void writeTo(final StreamOutput out) throws IOException {
        out.writeBoolean(binary);
    }

    /**
     * Converts a float[] vector to a little-endian byte array.
     * Each float is written as 4 bytes in little-endian format (native byte order on x86/ARM).
     * The returned byte[] is suitable for direct return from {@code nextValue()} —
     * XContentBuilder will base64-encode it during serialization.
     */
    public static byte[] floatToLittleEndianBytes(final float[] vector) {
        final ByteBuffer buffer = ByteBuffer.allocate(vector.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().put(vector);
        return buffer.array();
    }

    @Override
    public String toString() {
        return "knn_vector(" + formatName + ")";
    }
}
