/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BINARY;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

public class ResolvedIndexSpecBWCTests extends KNNTestCase {

    public void testOldStyleSQEncoderParams_resolvesCorrectly() {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(SQ_BITS, 1);
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> methodParams = new HashMap<>();
        methodParams.put(METHOD_ENCODER_PARAMETER, encoderCtx);

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, methodParams)
        );
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .versionCreated(Version.CURRENT)
            .mode(Mode.ON_DISK)
            .compressionLevel(CompressionLevel.x32)
            .build();

        KNNLibraryIndexingContext ctx = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, configContext);
        assertNotNull(ctx);
        ResolvedIndexSpec spec = ctx.getResolvedSpec();
        assertNotNull(spec);
        assertEquals(Encoder.EncoderType.SQ, spec.getEncoderType());
        assertEquals(Encoder.QuantizationBits.ONE, spec.getQuantizationBits());
    }

    public void testOldStyleBQEncoderParams_resolvesCorrectly() {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put("bits", 1);
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_BINARY, encoderParams);

        Map<String, Object> methodParams = new HashMap<>();
        methodParams.put(METHOD_ENCODER_PARAMETER, encoderCtx);

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, methodParams)
        );
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .versionCreated(Version.CURRENT)
            .mode(Mode.ON_DISK)
            .compressionLevel(CompressionLevel.x32)
            .build();

        KNNLibraryIndexingContext ctx = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, configContext);
        assertNotNull(ctx);
        ResolvedIndexSpec spec = ctx.getResolvedSpec();
        assertNotNull(spec);
        assertEquals(Encoder.EncoderType.BQ, spec.getEncoderType());
        assertEquals(Encoder.QuantizationBits.ONE, spec.getQuantizationBits());
    }

    public void testNoEncoder_defaultsToFlat() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, new HashMap<>())
        );
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .versionCreated(Version.CURRENT)
            .mode(Mode.NOT_CONFIGURED)
            .compressionLevel(CompressionLevel.x1)
            .build();

        KNNLibraryIndexingContext ctx = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, configContext);
        assertNotNull(ctx);
        ResolvedIndexSpec spec = ctx.getResolvedSpec();
        assertNotNull(spec);
        assertEquals(Encoder.EncoderType.FLAT, spec.getEncoderType());
        assertEquals(Encoder.QuantizationBits.FULL_PRECISION, spec.getQuantizationBits());
    }

    public void testSQWith16Bits_resolvesCorrectly() {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(SQ_BITS, 16);
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> methodParams = new HashMap<>();
        methodParams.put(METHOD_ENCODER_PARAMETER, encoderCtx);

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, methodParams)
        );
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .versionCreated(Version.CURRENT)
            .mode(Mode.NOT_CONFIGURED)
            .compressionLevel(CompressionLevel.x2)
            .build();

        KNNLibraryIndexingContext ctx = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, configContext);
        assertNotNull(ctx);
        ResolvedIndexSpec spec = ctx.getResolvedSpec();
        assertNotNull(spec);
        assertEquals(Encoder.EncoderType.SQ, spec.getEncoderType());
        assertEquals(Encoder.QuantizationBits.SIXTEEN, spec.getQuantizationBits());
    }

    public void testMethodComponentContextSerialization_unchanged() throws IOException {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(SQ_BITS, 1);
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> methodParams = new HashMap<>();
        methodParams.put(METHOD_ENCODER_PARAMETER, encoderCtx);
        MethodComponentContext original = new MethodComponentContext(METHOD_HNSW, methodParams);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        original.writeTo(streamOutput);

        MethodComponentContext deserialized = new MethodComponentContext(streamOutput.bytes().streamInput());
        assertEquals(original, deserialized);
        assertEquals(original.getName(), deserialized.getName());
        assertEquals(original.getParameters().size(), deserialized.getParameters().size());
    }
}
