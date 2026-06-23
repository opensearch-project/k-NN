/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
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

    /**
     * Sanity-checks the names users type in mappings; they are part of the method's public contract.
     */
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

    public void testSupportedEncoders_sqFlatLvq_notSvsPrefixed() {
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey(ENCODER_SQ));
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey(ENCODER_FLAT));
        assertTrue(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey(FAISS_SVS_ENCODER_LVQ));
        // Old svs_-prefixed encoder names must no longer be registered.
        assertFalse(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey("svs_fp16"));
        assertFalse(FaissSVSVamanaMethod.SUPPORTED_ENCODERS.containsKey("svs_sq8"));
        assertEquals(3, FaissSVSVamanaMethod.SUPPORTED_ENCODERS.size());
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
}
