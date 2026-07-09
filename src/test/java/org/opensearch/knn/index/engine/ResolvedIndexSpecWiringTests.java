/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class ResolvedIndexSpecWiringTests extends KNNTestCase {

    public void testKNNLibraryIndexingContextImpl_resolvedSpecStoredAndRetrieved() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();

        KNNLibraryIndexingContext ctx = KNNLibraryIndexingContextImpl.builder().resolvedSpec(spec).build();

        assertNotNull(ctx.getResolvedSpec());
        assertSame(spec, ctx.getResolvedSpec());
        assertEquals(KNNEngine.FAISS, ctx.getResolvedSpec().getEngine());
        assertEquals(Encoder.EncoderType.SQ, ctx.getResolvedSpec().getEncoderType());
        assertEquals(Encoder.QuantizationBits.ONE, ctx.getResolvedSpec().getQuantizationBits());
    }

    public void testKNNLibraryIndexingContextImpl_resolvedSpecDefaultsToNull() {
        KNNLibraryIndexingContext ctx = KNNLibraryIndexingContextImpl.builder().build();
        assertNull(ctx.getResolvedSpec());
    }

    public void testKNNLibraryIndexingContext_defaultMethodReturnsNull() {
        KNNLibraryIndexingContext ctx = new KNNLibraryIndexingContext() {
            @Override
            public java.util.Map<String, Object> getLibraryParameters() {
                return null;
            }

            @Override
            public org.opensearch.knn.index.engine.qframe.QuantizationConfig getQuantizationConfig() {
                return null;
            }

            @Override
            public org.opensearch.knn.index.mapper.VectorValidator getVectorValidator() {
                return null;
            }

            @Override
            public org.opensearch.knn.index.mapper.PerDimensionValidator getPerDimensionValidator() {
                return null;
            }

            @Override
            public org.opensearch.knn.index.mapper.PerDimensionProcessor getPerDimensionProcessor() {
                return null;
            }

            @Override
            public
                java.util.function.Function<TrainingConfigValidationInput, TrainingConfigValidationOutput>
                getTrainingConfigValidationSetup() {
                return null;
            }

            @Override
            public org.opensearch.knn.index.mapper.VectorTransformer getVectorTransformer() {
                return null;
            }

            @Override
            public ResolvedIndexSpec getResolvedSpec() {
                return null;
            }
        };
        assertNull(ctx.getResolvedSpec());
    }

    public void testFaissHNSWResolution_producesSpecInContext() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, new java.util.HashMap<>())
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
        assertEquals(KNNEngine.FAISS, spec.getEngine());
        assertEquals(METHOD_HNSW, spec.getMethodName());
        assertEquals(Encoder.EncoderType.FLAT, spec.getEncoderType());
        assertEquals(Encoder.QuantizationBits.FULL_PRECISION, spec.getQuantizationBits());
        assertEquals(CompressionLevel.x1, spec.getCompressionLevel());
        assertEquals(VectorDataType.FLOAT, spec.getVectorDataType());
        assertEquals(128, spec.getDimension());
    }

    public void testLuceneHNSWResolution_producesSpecInContext() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, new java.util.HashMap<>())
        );
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(256)
            .versionCreated(Version.CURRENT)
            .mode(Mode.NOT_CONFIGURED)
            .compressionLevel(CompressionLevel.x1)
            .build();

        KNNLibraryIndexingContext ctx = KNNEngine.LUCENE.getKNNLibraryIndexingContext(knnMethodContext, configContext);
        assertNotNull(ctx);
        ResolvedIndexSpec spec = ctx.getResolvedSpec();
        assertNotNull(spec);
        assertEquals(KNNEngine.LUCENE, spec.getEngine());
        assertEquals(METHOD_HNSW, spec.getMethodName());
        assertEquals(Encoder.EncoderType.FLAT, spec.getEncoderType());
        assertEquals(256, spec.getDimension());
    }

    public void testFaissHNSWWithSQEncoder_producesCorrectSpec() {
        java.util.Map<String, Object> encoderParams = new java.util.HashMap<>();
        encoderParams.put("bits", 1);
        MethodComponentContext encoderCtx = new MethodComponentContext("sq", encoderParams);

        java.util.Map<String, Object> methodParams = new java.util.HashMap<>();
        methodParams.put("encoder", encoderCtx);

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, methodParams)
        );
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(768)
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
        assertEquals(CompressionLevel.x32, spec.getCompressionLevel());
        assertEquals(Mode.ON_DISK, spec.getMode());
        assertEquals(768, spec.getDimension());
    }

    public void testNullDimensionHandledGracefully() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, new java.util.HashMap<>())
        );
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .versionCreated(Version.CURRENT)
            .mode(Mode.NOT_CONFIGURED)
            .compressionLevel(CompressionLevel.x1)
            .build();

        KNNLibraryIndexingContext ctx = KNNEngine.FAISS.getKNNLibraryIndexingContext(knnMethodContext, configContext);
        assertNotNull(ctx);
        ResolvedIndexSpec spec = ctx.getResolvedSpec();
        assertNotNull(spec);
        assertEquals(0, spec.getDimension());
    }

    public void testFaissHNSWWithBQEncoder_defaultsToOneBit() {
        java.util.Map<String, Object> encoderParams = new java.util.HashMap<>();
        MethodComponentContext encoderCtx = new MethodComponentContext("binary", encoderParams);

        java.util.Map<String, Object> methodParams = new java.util.HashMap<>();
        methodParams.put("encoder", encoderCtx);

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
        ResolvedIndexSpec spec = ctx.getResolvedSpec();
        assertEquals(Encoder.EncoderType.BQ, spec.getEncoderType());
        assertEquals(Encoder.QuantizationBits.ONE, spec.getQuantizationBits());
    }
}
