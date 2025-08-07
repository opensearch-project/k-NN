/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.util;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;

import java.io.IOException;

/**
 * Utility class for quantization-related operations.
 */
public final class QuantizationUtils {

    private QuantizationUtils() {
        // Utility class - prevent instantiation
    }

    /**
     * Wrapper class for float arrays to enable serialization via writeOptionalArray/readOptionalArray.
     * This is needed because OpenSearch's optional array serialization requires Writeable objects.
     */
    @Getter
    @AllArgsConstructor
    public static class FloatArrayWrapper implements Writeable {
        private final float[] array;

        public FloatArrayWrapper(StreamInput in) throws IOException {
            this.array = in.readFloatArray();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeFloatArray(array);
        }
    }
}
