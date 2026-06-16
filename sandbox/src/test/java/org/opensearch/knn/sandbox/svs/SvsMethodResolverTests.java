/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE_FP16;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_PRIMARY_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_RESIDUAL_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_SVS_VAMANA;

public class SvsMethodResolverTests extends OpenSearchTestCase {

    private static final SvsMethodResolver RESOLVER = new SvsMethodResolver();

    private KNNMethodConfigContext.KNNMethodConfigContextBuilder configBuilder() {
        return KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).dimension(128).versionCreated(Version.CURRENT);
    }

    private KNNMethodContext svsVamanaContext() {
        return new KNNMethodContext(KNNEngine.EXPERIMENTAL, SpaceType.L2, new MethodComponentContext(METHOD_SVS_VAMANA, Map.of()));
    }

    public void testResolveMethod_whenNoMethodContext_thenDefaultsToSvsVamana() {
        ResolvedMethodContext resolved = RESOLVER.resolveMethod(null, configBuilder().build(), false, SpaceType.L2);
        assertEquals(METHOD_SVS_VAMANA, resolved.getKnnMethodContext().getMethodComponentContext().getName());
        assertEquals(CompressionLevel.x1, resolved.getCompressionLevel());
    }

    public void testResolveMethod_whenOnDisk_thenThrow() {
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> RESOLVER.resolveMethod(svsVamanaContext(), configBuilder().mode(Mode.ON_DISK).build(), false, SpaceType.L2)
        );
        assertTrue(e.getMessage().contains("mode=on_disk is not supported with svs_vamana"));
    }

    public void testResolveMethod_whenTrainingContext_thenThrow() {
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> RESOLVER.resolveMethod(svsVamanaContext(), configBuilder().build(), true, SpaceType.L2)
        );
        assertTrue(e.getMessage().contains("training"));
    }

    public void testResolveMethod_whenWrongMethodName_thenThrow() {
        KNNMethodContext hnswOnSvs = new KNNMethodContext(
            KNNEngine.EXPERIMENTAL,
            SpaceType.L2,
            new MethodComponentContext("hnsw", Map.of())
        );
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> RESOLVER.resolveMethod(hnswOnSvs, configBuilder().build(), false, SpaceType.L2)
        );
        assertTrue(e.getMessage().contains("Invalid method name"));
    }

    public void testResolveMethod_whenCompressionX2_thenSqFp16() {
        ResolvedMethodContext resolved = RESOLVER.resolveMethod(
            null,
            configBuilder().compressionLevel(CompressionLevel.x2).build(),
            false,
            SpaceType.L2
        );
        MethodComponentContext encoder = (MethodComponentContext) resolved.getKnnMethodContext()
            .getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        assertEquals(ENCODER_SQ, encoder.getName());
        assertEquals(FAISS_SVS_SQ_TYPE_FP16, encoder.getParameters().get(FAISS_SVS_SQ_TYPE));
        assertEquals(CompressionLevel.x2, resolved.getCompressionLevel());
    }

    public void testResolveMethod_whenCompressionX4_thenLvq4x4() {
        ResolvedMethodContext resolved = RESOLVER.resolveMethod(
            null,
            configBuilder().compressionLevel(CompressionLevel.x4).build(),
            false,
            SpaceType.L2
        );
        MethodComponentContext encoder = (MethodComponentContext) resolved.getKnnMethodContext()
            .getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        assertEquals(FAISS_SVS_ENCODER_LVQ, encoder.getName());
        assertEquals(4, encoder.getParameters().get(METHOD_PARAMETER_LVQ_PRIMARY_BITS));
        assertEquals(4, encoder.getParameters().get(METHOD_PARAMETER_LVQ_RESIDUAL_BITS));
        assertEquals(CompressionLevel.x4, resolved.getCompressionLevel());
    }

    public void testResolveMethod_whenCompressionX8_thenLvq4x0() {
        ResolvedMethodContext resolved = RESOLVER.resolveMethod(
            null,
            configBuilder().compressionLevel(CompressionLevel.x8).build(),
            false,
            SpaceType.L2
        );
        MethodComponentContext encoder = (MethodComponentContext) resolved.getKnnMethodContext()
            .getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        assertEquals(FAISS_SVS_ENCODER_LVQ, encoder.getName());
        assertEquals(4, encoder.getParameters().get(METHOD_PARAMETER_LVQ_PRIMARY_BITS));
        assertEquals(0, encoder.getParameters().get(METHOD_PARAMETER_LVQ_RESIDUAL_BITS));
        assertEquals(CompressionLevel.x8, resolved.getCompressionLevel());
    }

    public void testResolveMethod_whenUnsupportedCompression_thenThrow() {
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> RESOLVER.resolveMethod(
                svsVamanaContext(),
                configBuilder().compressionLevel(CompressionLevel.x16).build(),
                false,
                SpaceType.L2
            )
        );
        assertTrue(e.getMessage().contains("compression"));
    }
}
