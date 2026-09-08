/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_ALPHA;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_DEGREE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_PRIMARY_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_RESIDUAL_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_SVS_VAMANA;

public class FaissSVSVamanaMethodTests extends OpenSearchTestCase {

    private KNNMethodConfigContext configContext() {
        return KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).vectorDataType(VectorDataType.FLOAT).dimension(128).build();
    }

    public void testPublicNames_methodAndEncoderParameters() {
        assertEquals("svs_vamana", METHOD_SVS_VAMANA);
        assertEquals("lvq", FAISS_SVS_ENCODER_LVQ);
        assertEquals("primary_bits", METHOD_PARAMETER_LVQ_PRIMARY_BITS);
        assertEquals("residual_bits", METHOD_PARAMETER_LVQ_RESIDUAL_BITS);
    }

    public void testSupportedSpaces_includesL2IPCosine() {
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_SPACES.contains(SpaceType.L2));
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_SPACES.contains(SpaceType.INNER_PRODUCT));
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_SPACES.contains(SpaceType.COSINESIMIL));
    }

    public void testSupportedEncoders_sqFlatLvqLeanVec_notSvsPrefixed() {
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey(ENCODER_SQ));
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey(ENCODER_FLAT));
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey(FAISS_SVS_ENCODER_LVQ));
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey(SVSConstants.FAISS_SVS_ENCODER_LEANVEC));
        assertFalse(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey("svs_fp16"));
        assertFalse(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey("svs_sq8"));
        assertEquals(4, FaissSVSVamanaMethod.SUPPORTED_ENCODERS.size());
    }

    public void testParametersPresent_degreeConstructionAlpha() {
        MethodComponent component = FaissSVSVamanaMethod.METHOD_COMPONENT;
        assertTrue(component.getParameters().containsKey(METHOD_PARAMETER_DEGREE));
        assertTrue(component.getParameters().containsKey(METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE));
        assertTrue(component.getParameters().containsKey(METHOD_PARAMETER_ALPHA));
    }

    public void testDegreeValidation_bounds() {
        MethodComponent component = FaissSVSVamanaMethod.METHOD_COMPONENT;
        assertNull(component.validate(new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_DEGREE, 64)), configContext()));
        ValidationException tooLow = component.validate(
            new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_DEGREE, 0)),
            configContext()
        );
        assertNotNull(tooLow);
        ValidationException tooHigh = component.validate(
            new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_DEGREE, 257)),
            configContext()
        );
        assertNotNull(tooHigh);
    }

    public void testConstructionWindowValidation_mustBePositive() {
        MethodComponent component = FaissSVSVamanaMethod.METHOD_COMPONENT;
        assertNull(
            component.validate(
                new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE, 200)),
                configContext()
            )
        );
        assertNotNull(
            component.validate(
                new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE, 0)),
                configContext()
            )
        );
    }

    public void testAlphaValidation_mustBePositive() {
        MethodComponent component = FaissSVSVamanaMethod.METHOD_COMPONENT;
        assertNull(component.validate(new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_ALPHA, 1.2)), configContext()));
        assertNotNull(
            component.validate(new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_ALPHA, 0.0)), configContext())
        );
        assertNotNull(
            component.validate(new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_ALPHA, -1.0)), configContext())
        );
    }

    private String indexDescriptionFor(MethodComponentContext methodComponentContext) {
        KNNLibraryIndexingContext indexingContext = FaissSVSVamanaMethod.METHOD_COMPONENT.getKNNLibraryIndexingContext(
            methodComponentContext,
            configContext()
        );
        return (String) indexingContext.getLibraryParameters().get(INDEX_DESCRIPTION_PARAMETER);
    }

    public void testIndexDescription_whenDefaultFlatEncoder_thenTrailingFlatDropped() {
        assertEquals(
            "SVSVamana64",
            indexDescriptionFor(new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_DEGREE, 64)))
        );
    }

    public void testIndexDescription_whenLvqEncoder_thenBitsSuffixed() {
        MethodComponentContext lvq = new MethodComponentContext(
            FAISS_SVS_ENCODER_LVQ,
            Map.of(METHOD_PARAMETER_LVQ_PRIMARY_BITS, 4, METHOD_PARAMETER_LVQ_RESIDUAL_BITS, 4)
        );
        try {
            assertEquals(
                "SVSVamana64,LVQ4x4",
                indexDescriptionFor(
                    new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_DEGREE, 64, METHOD_ENCODER_PARAMETER, lvq))
                )
            );
        } catch (ExceptionInInitializerError | NoClassDefFoundError e) {
            // The generator runs the native platform check; skip without the library (the IT covers it).
            org.junit.Assume.assumeNoException("LVQ platform check needs the SVS native library", e);
        } catch (IllegalArgumentException e) {
            if (e.getCause() instanceof LinkageError == false) {
                throw e;
            }
            org.junit.Assume.assumeNoException("LVQ platform check needs the SVS native library", e);
        }
    }

    public void testIndexDescription_whenLeanVecEncoder_thenBitsAndDimsSuffixed() {
        MethodComponentContext leanVec = new MethodComponentContext(
            SVSConstants.FAISS_SVS_ENCODER_LEANVEC,
            Map.of(
                METHOD_PARAMETER_LVQ_PRIMARY_BITS,
                4,
                METHOD_PARAMETER_LVQ_RESIDUAL_BITS,
                8,
                SVSConstants.METHOD_PARAMETER_LEANVEC_DIMENSIONS,
                192
            )
        );
        try {
            assertEquals(
                "SVSVamana64,LeanVec4x8_192",
                indexDescriptionFor(
                    new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_DEGREE, 64, METHOD_ENCODER_PARAMETER, leanVec))
                )
            );
        } catch (ExceptionInInitializerError | NoClassDefFoundError e) {
            org.junit.Assume.assumeNoException("LeanVec platform check needs the SVS native library", e);
        } catch (IllegalArgumentException e) {
            if (e.getCause() instanceof LinkageError == false) {
                throw e;
            }
            org.junit.Assume.assumeNoException("LeanVec platform check needs the SVS native library", e);
        }
    }

    public void testIndexDescription_whenLeanVecEncoderNoDims_thenNoSuffix() {
        MethodComponentContext leanVec = new MethodComponentContext(SVSConstants.FAISS_SVS_ENCODER_LEANVEC, Map.of());
        try {
            assertEquals(
                "SVSVamana64,LeanVec4x8",
                indexDescriptionFor(
                    new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_DEGREE, 64, METHOD_ENCODER_PARAMETER, leanVec))
                )
            );
        } catch (ExceptionInInitializerError | NoClassDefFoundError e) {
            org.junit.Assume.assumeNoException("LeanVec platform check needs the SVS native library", e);
        } catch (IllegalArgumentException e) {
            if (e.getCause() instanceof LinkageError == false) {
                throw e;
            }
            org.junit.Assume.assumeNoException("LeanVec platform check needs the SVS native library", e);
        }
    }

    public void testIndexDescription_whenSqFp16Encoder_thenFp16Token() {
        MethodComponentContext sq = new MethodComponentContext(ENCODER_SQ, Map.of(SVSConstants.FAISS_SVS_SQ_TYPE, "fp16"));
        assertEquals(
            "SVSVamana64,FP16",
            indexDescriptionFor(
                new MethodComponentContext(METHOD_SVS_VAMANA, Map.of(METHOD_PARAMETER_DEGREE, 64, METHOD_ENCODER_PARAMETER, sq))
            )
        );
    }

    /**
     * Every encoder name must survive index-spec resolution (the core default rejects lvq/leanvec).
     */
    public void testIndexingContext_resolvesForEveryEncoder() {
        for (String encoderName : new String[] { ENCODER_FLAT, ENCODER_SQ }) {
            ResolvedIndexSpec spec = new FaissSVSVamanaMethod().getKNNLibraryIndexingContext(
                methodContextWithEncoder(encoderName),
                configContext()
            ).getResolvedSpec();
            assertNotNull(encoderName, spec);
            Encoder.EncoderType expectedType = ENCODER_FLAT.equals(encoderName) ? Encoder.EncoderType.FLAT : Encoder.EncoderType.SQ;
            assertEquals(encoderName, expectedType, spec.getEncoderType());
            assertFalse(encoderName, spec.isMemoryOptimizedEligible());
        }
        for (String encoderName : new String[] { FAISS_SVS_ENCODER_LVQ, "leanvec" }) {
            ResolvedIndexSpec spec = FaissSVSVamanaMethod.buildSvsResolvedSpec(methodContextWithEncoder(encoderName), configContext());
            assertNotNull(encoderName, spec);
            assertEquals(METHOD_SVS_VAMANA, spec.getMethodName());
            assertEquals(encoderName, Encoder.EncoderType.SQ, spec.getEncoderType());
            assertFalse(encoderName, spec.isMemoryOptimizedEligible());
        }
    }

    private static KNNMethodContext methodContextWithEncoder(String encoderName) {
        return new KNNMethodContext(
            KNNEngine.getEngine(SVSConstants.SVS_ENGINE_NAME),
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_SVS_VAMANA,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(encoderName, Map.of()))
            )
        );
    }

}
