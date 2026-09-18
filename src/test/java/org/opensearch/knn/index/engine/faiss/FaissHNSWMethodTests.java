/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.io.IOException;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class FaissHNSWMethodTests extends KNNTestCase {

    public void testValidate_whenHalfFloatWithNoEncoder_thenAccepted() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.HALF_FLOAT)
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject();
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(xContentBuilderToMap(xContentBuilder));

        assertNull(new FaissHNSWMethod().validate(knnMethodContext, knnMethodConfigContext));
    }

    public void testValidate_whenHalfFloatWithSqBits16Encoder_thenRejected() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.HALF_FLOAT)
            .build();

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16)))
            )
        );

        assertNotNull(new FaissHNSWMethod().validate(knnMethodContext, knnMethodConfigContext));
    }

    public void testValidate_whenHalfFloatWithSqBits1Encoder_thenAccepted() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.HALF_FLOAT)
            .build();

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)))
            )
        );

        assertNull(new FaissHNSWMethod().validate(knnMethodContext, knnMethodConfigContext));
    }

    public void testSupportedEncoders_containsFlatSqPqAndQFrame() {
        Map<String, Encoder> encoders = FaissHNSWMethod.SUPPORTED_ENCODERS;
        assertTrue(encoders.containsKey(ENCODER_FLAT));
        assertTrue(encoders.containsKey(ENCODER_SQ));
        assertTrue(encoders.containsKey(ENCODER_PQ));
        assertTrue(encoders.containsKey(QFrameBitEncoder.NAME));
        assertEquals(4, encoders.size());
    }

    public void testSupportedEncoders_sqEncoderIsFaissSQEncoder() {
        Encoder sqEncoder = FaissHNSWMethod.SUPPORTED_ENCODERS.get(ENCODER_SQ);
        assertNotNull(sqEncoder);
        assertTrue(sqEncoder instanceof FaissSQEncoder);
    }

    public void testSupportedEncoders_flatEncoderCompressionIsX1() {
        Encoder flatEncoder = FaissHNSWMethod.SUPPORTED_ENCODERS.get(ENCODER_FLAT);
        assertEquals(CompressionLevel.x1, flatEncoder.calculateCompressionLevel(null, null));
    }

    public void testSupportedEncoders_sqEncoderBits1CompressionIsX32() {
        Encoder sqEncoder = FaissHNSWMethod.SUPPORTED_ENCODERS.get(ENCODER_SQ);
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1));
        assertEquals(CompressionLevel.x32, sqEncoder.calculateCompressionLevel(mcc, null));
    }

    public void testSupportedEncoders_sqEncoderBits16CompressionIsX2() {
        Encoder sqEncoder = FaissHNSWMethod.SUPPORTED_ENCODERS.get(ENCODER_SQ);
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16));
        assertEquals(CompressionLevel.x2, sqEncoder.calculateCompressionLevel(mcc, null));
    }

    public void testIsSQOneBitIndex_whenSQWithBits1Float_thenTrue() {
        assertSQOneBitIndex(VectorDataType.FLOAT, ENCODER_SQ, Map.of(SQ_BITS, 1), true);
    }

    public void testIsSQOneBitIndex_whenSQWithBits16Float_thenFalse() {
        assertSQOneBitIndex(VectorDataType.FLOAT, ENCODER_SQ, Map.of(SQ_BITS, 16), false);
    }

    public void testIsSQOneBitIndex_whenFlatEncoderFloat_thenFalse() {
        assertSQOneBitIndex(VectorDataType.FLOAT, ENCODER_FLAT, Map.of(), false);
    }

    public void testIsSQOneBitIndex_whenSQWithBits1Binary_thenFalse() {
        assertSQOneBitIndex(VectorDataType.BINARY, ENCODER_SQ, Map.of(SQ_BITS, 1), false);
    }

    public void testIsSQOneBitIndex_whenSQWithBits1Byte_thenFalse() {
        assertSQOneBitIndex(VectorDataType.BYTE, ENCODER_SQ, Map.of(SQ_BITS, 1), false);
    }

    public void testIsSQOneBitIndex_whenNoBitsParam_thenFalse() {
        assertSQOneBitIndex(VectorDataType.FLOAT, ENCODER_SQ, Map.of(), false);
    }

    public void testIsFloat16Index_whenSQWithBits16Float_thenTrue() {
        assertIsFloat16Index(VectorDataType.FLOAT, ENCODER_SQ, Map.of(SQ_BITS, 16), true);
    }

    public void testIsFloat16Index_whenSQLegacyNoBitsFloat_thenTrue() {
        assertIsFloat16Index(VectorDataType.FLOAT, ENCODER_SQ, Map.of(), true);
    }

    public void testIsFloat16Index_whenSQWithBits1Float_thenFalse() {
        assertIsFloat16Index(VectorDataType.FLOAT, ENCODER_SQ, Map.of(SQ_BITS, 1), false);
    }

    public void testIsFloat16Index_whenFlatEncoder_thenFalse() {
        assertIsFloat16Index(VectorDataType.FLOAT, ENCODER_FLAT, Map.of(), false);
    }

    public void testIsFloat16Index_whenBinaryDataType_thenFalse() {
        assertIsFloat16Index(VectorDataType.BINARY, ENCODER_SQ, Map.of(SQ_BITS, 16), false);
    }

    public void testTrainingConfigValidation_whenHNSWWithSQBits1_thenValid() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .compressionLevel(CompressionLevel.x32)
            .build();

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)))
            )
        );

        KNNLibraryIndexingContext indexingContext = method.getKNNLibraryIndexingContext(methodContext, configContext);
        Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> validationSetup = indexingContext
            .getTrainingConfigValidationSetup();

        TrainingConfigValidationOutput output = validationSetup.apply(
            TrainingConfigValidationInput.builder().knnMethodContext(methodContext).knnMethodConfigContext(configContext).build()
        );
        assertNull(output.getValid());
    }

    public void testTrainingConfigValidation_whenHNSWWithSQBits1AndX2Compression_thenInvalid() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .compressionLevel(CompressionLevel.x2)
            .build();

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)))
            )
        );

        KNNLibraryIndexingContext indexingContext = method.getKNNLibraryIndexingContext(methodContext, configContext);
        Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> validationSetup = indexingContext
            .getTrainingConfigValidationSetup();

        TrainingConfigValidationOutput output = validationSetup.apply(
            TrainingConfigValidationInput.builder().knnMethodContext(methodContext).knnMethodConfigContext(configContext).build()
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("incompatible"));
    }

    public void testTrainingConfigValidation_whenHNSWWithNoEncoder_thenValid() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );

        KNNLibraryIndexingContext indexingContext = method.getKNNLibraryIndexingContext(methodContext, configContext);
        Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> validationSetup = indexingContext
            .getTrainingConfigValidationSetup();

        TrainingConfigValidationOutput output = validationSetup.apply(
            TrainingConfigValidationInput.builder().knnMethodContext(methodContext).knnMethodConfigContext(configContext).build()
        );
        assertNull(output.getValid());
    }

    public void testDefaultEncoder_isFlatEncoder() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );

        Map<String, Object> libraryParams = method.getKNNLibraryIndexingContext(methodContext, configContext).getLibraryParameters();
        @SuppressWarnings("unchecked")
        Map<String, Object> innerParams = (Map<String, Object>) libraryParams.get(PARAMETERS);
        @SuppressWarnings("unchecked")
        Map<String, Object> encoderParams = (Map<String, Object>) innerParams.get(METHOD_ENCODER_PARAMETER);
        assertEquals(ENCODER_FLAT, encoderParams.get(NAME));
    }

    // --- supportsRemoteIndexBuild: half_float (#3575) ---

    public void testSupportsRemoteIndexBuild_whenHalfFloatNoEncoder_thenSupported() {
        // x1: flat fp16 native storage, no encoder resolved - uploaded as fp32, converted by the service
        Map<String, Object> params = Map.of(
            NAME,
            METHOD_HNSW,
            VECTOR_DATA_TYPE_FIELD,
            VectorDataType.HALF_FLOAT.getValue(),
            PARAMETERS,
            Map.of()
        );
        assertTrue(FaissHNSWMethod.supportsRemoteIndexBuild(params));
    }

    public void testSupportsRemoteIndexBuild_whenHalfFloatSQOneBit_thenSupported() {
        // x16 resolves internally to sq bits=1 - the 1-bit codes upload unchanged
        Map<String, Object> params = buildLibraryParametersMap(VectorDataType.HALF_FLOAT, ENCODER_SQ, Map.of(SQ_BITS, 1));
        assertTrue(FaissHNSWMethod.supportsRemoteIndexBuild(params));
    }

    public void testSupportsRemoteIndexBuild_whenHalfFloatSQSixteenBit_thenNotSupported() {
        // fp16 SQ on half_float is not a valid mapping; the gate must not admit it either
        Map<String, Object> params = buildLibraryParametersMap(VectorDataType.HALF_FLOAT, ENCODER_SQ, Map.of(SQ_BITS, 16));
        assertFalse(FaissHNSWMethod.supportsRemoteIndexBuild(params));
    }

    public void testSupportsRemoteIndexBuild_whenFloatConfigs_thenUnchanged() {
        // Regression guard: FLOAT eligibility unchanged by the half_float gate
        assertTrue(FaissHNSWMethod.supportsRemoteIndexBuild(buildLibraryParametersMap(VectorDataType.FLOAT, ENCODER_FLAT, Map.of())));
        assertTrue(
            FaissHNSWMethod.supportsRemoteIndexBuild(buildLibraryParametersMap(VectorDataType.FLOAT, ENCODER_SQ, Map.of(SQ_BITS, 1)))
        );
        assertFalse(FaissHNSWMethod.supportsRemoteIndexBuild(buildLibraryParametersMap(VectorDataType.FLOAT, ENCODER_PQ, Map.of())));
    }

    private void assertSQOneBitIndex(VectorDataType dataType, String encoderName, Map<String, Object> encoderParams, boolean expected) {
        Map<String, Object> params = buildLibraryParametersMap(dataType, encoderName, encoderParams);
        assertEquals(expected, FaissHNSWMethod.isSQOneBitIndex(dataType, params));
    }

    private void assertIsFloat16Index(VectorDataType dataType, String encoderName, Map<String, Object> encoderParams, boolean expected) {
        Map<String, Object> params = buildLibraryParametersMap(dataType, encoderName, encoderParams);
        assertEquals(expected, FaissHNSWMethod.isFloat16Index(dataType, params));
    }

    private Map<String, Object> buildLibraryParametersMap(
        VectorDataType vectorDataType,
        String encoderName,
        Map<String, Object> encoderParams
    ) {
        Map<String, Object> encoder = new java.util.HashMap<>(encoderParams);
        encoder.put(NAME, encoderName);
        return Map.of(
            NAME,
            METHOD_HNSW,
            VECTOR_DATA_TYPE_FIELD,
            vectorDataType.getValue(),
            PARAMETERS,
            Map.of(METHOD_ENCODER_PARAMETER, encoder)
        );
    }
}
