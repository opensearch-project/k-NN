/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;

public class AbstractFaissPQEncoderTests extends KNNTestCase {

    @SneakyThrows
    public void testCalculateCompressionLevel() {
        AbstractFaissPQEncoder encoder = new AbstractFaissPQEncoder() {
            @Override
            public MethodComponent getMethodComponent() {
                return FaissIVFPQEncoder.METHOD_COMPONENT;
            }
        };

        // Compression formula is:
        // actual_compression = (d*32)/(m*code_size) and then round down to nearest: 1x, 2x, 4x, 8x, 16x, 32x

        // d=768
        // m=2
        // code_size=8
        // actual_compression = (768*32)/(2*8) = 1,536x
        // expected_compression = Max compression level
        assertCompressionLevel(2, 8, 768, CompressionLevel.MAX_COMPRESSION_LEVEL, encoder);

        // d=32
        // m=4
        // code_size=16
        // actual_compression = (32*32)/(4*16) = 16x
        // expected_compression = Max compression level
        assertCompressionLevel(4, 16, 32, CompressionLevel.x16, encoder);

        // d=1536
        // m=768
        // code_size=8
        // actual_compression = (1536*32)/(768*8) = 8x
        // expected_compression = Max compression level
        assertCompressionLevel(768, 8, 1536, CompressionLevel.x8, encoder);

        // d=128
        // m=128
        // code_size=8
        // actual_compression = (128*32)/(128*8) = 4x
        // expected_compression = Max compression level
        assertCompressionLevel(128, 8, 128, CompressionLevel.x4, encoder);
    }

    private void assertCompressionLevel(int m, int codeSize, int d, CompressionLevel expectedCompression, Encoder encoder) {
        assertEquals(
            expectedCompression,
            encoder.calculateCompressionLevel(generateMethodComponentContext(m, codeSize), generateKNNMethodConfigContext(d))
        );
    }

    private MethodComponentContext generateMethodComponentContext(int m, int codeSize) {
        return new MethodComponentContext(ENCODER_PQ, Map.of(ENCODER_PARAMETER_PQ_M, m, ENCODER_PARAMETER_PQ_CODE_SIZE, codeSize));
    }

    private KNNMethodConfigContext generateKNNMethodConfigContext(int dimension) {
        return KNNMethodConfigContext.builder().dimension(dimension).build();
    }
}
