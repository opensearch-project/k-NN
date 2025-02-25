/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;

public class FaissMethodResolverTests extends KNNTestCase {

    MethodResolver TEST_RESOLVER = new FaissMethodResolver();

    public void testResolveMethod_whenValid_thenResolve() {
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x1, SpaceType.INNER_PRODUCT, ENCODER_FLAT, false);

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .mode(Mode.ON_DISK)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x32, SpaceType.INNER_PRODUCT, QFrameBitEncoder.NAME, true);

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .mode(Mode.ON_DISK)
                .compressionLevel(CompressionLevel.x16)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x16, SpaceType.INNER_PRODUCT, QFrameBitEncoder.NAME, true);

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .compressionLevel(CompressionLevel.x16)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x16, SpaceType.INNER_PRODUCT, QFrameBitEncoder.NAME, true);

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            new KNNMethodContext(
                KNNEngine.FAISS,
                SpaceType.L2,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(
                        METHOD_ENCODER_PARAMETER,
                        new MethodComponentContext(
                            QFrameBitEncoder.NAME,
                            Map.of(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x8.numBitsForFloat32())
                        )
                    )
                )
            ),
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .mode(Mode.ON_DISK)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.L2
        );
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x8, SpaceType.L2, QFrameBitEncoder.NAME, true);

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            new KNNMethodContext(
                KNNEngine.FAISS,
                SpaceType.L2,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(
                        METHOD_ENCODER_PARAMETER,
                        new MethodComponentContext(
                            QFrameBitEncoder.NAME,
                            Map.of(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x8.numBitsForFloat32())
                        )
                    )
                )
            ),
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.L2
        );
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x8, SpaceType.L2, QFrameBitEncoder.NAME, true);

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, Map.of())),
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.L2
        );
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x1, SpaceType.L2, ENCODER_FLAT, false);

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, Map.of())),
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.BINARY).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.L2
        );
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x1, SpaceType.L2, ENCODER_FLAT, false);

        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .versionCreated(Version.CURRENT)
            .build();

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            new KNNMethodContext(
                KNNEngine.FAISS,
                SpaceType.L2,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(
                        METHOD_ENCODER_PARAMETER,
                        new MethodComponentContext(
                            QFrameBitEncoder.NAME,
                            Map.of(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x8.numBitsForFloat32())
                        )
                    )
                )
            ),
            knnMethodConfigContext,
            false,
            SpaceType.L2
        );
        assertEquals(knnMethodConfigContext.getCompressionLevel(), CompressionLevel.x8);
        validateResolveMethodContext(resolvedMethodContext, CompressionLevel.x8, SpaceType.L2, QFrameBitEncoder.NAME, true);
    }

    private void validateResolveMethodContext(
        ResolvedMethodContext resolvedMethodContext,
        CompressionLevel expectedCompression,
        SpaceType expectedSpaceType,
        String expectedEncoderName,
        boolean checkBitsEncoderParam
    ) {
        assertEquals(expectedCompression, resolvedMethodContext.getCompressionLevel());
        assertEquals(KNNEngine.FAISS, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(expectedSpaceType, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(
            expectedEncoderName,
            ((MethodComponentContext) resolvedMethodContext.getKnnMethodContext()
                .getMethodComponentContext()
                .getParameters()
                .get(METHOD_ENCODER_PARAMETER)).getName()
        );
        if (checkBitsEncoderParam) {
            assertEquals(
                expectedCompression.numBitsForFloat32(),
                ((MethodComponentContext) resolvedMethodContext.getKnnMethodContext()
                    .getMethodComponentContext()
                    .getParameters()
                    .get(METHOD_ENCODER_PARAMETER)).getParameters().get(QFrameBitEncoder.BITCOUNT_PARAM)
            );
        }

    }

    public void testResolveMethod_whenInvalid_thenThrow() {
        // Invalid compression
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .compressionLevel(CompressionLevel.x4)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );

        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.BINARY)
                    .compressionLevel(CompressionLevel.x4)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );

        // Invalid spec ondisk and compression is 1
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .mode(Mode.ON_DISK)
                    .compressionLevel(CompressionLevel.x1)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );

        // Invalid compression conflict
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                new KNNMethodContext(
                    KNNEngine.FAISS,
                    SpaceType.INNER_PRODUCT,
                    new MethodComponentContext(
                        METHOD_HNSW,
                        Map.of(
                            METHOD_ENCODER_PARAMETER,
                            new MethodComponentContext(
                                QFrameBitEncoder.NAME,
                                Map.of(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x32.numBitsForFloat32())
                            )
                        )
                    )
                ),
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .mode(Mode.ON_DISK)
                    .compressionLevel(CompressionLevel.x8)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.INNER_PRODUCT
            )

        );

        Map<String, Object> parameters = Map.of(
            ENCODER_PARAMETER_PQ_M,
            3,
            METHOD_ENCODER_PARAMETER,
            new MethodComponentContext("pq", Map.of())
        );
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, parameters);
        final KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, SpaceType.INNER_PRODUCT, methodComponentContext);

        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(10)
            .versionCreated(Version.CURRENT)
            .compressionLevel(CompressionLevel.x8)
            .mode(Mode.ON_DISK)
            .build();

        ValidationException validationException = expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(knnMethodContext, knnMethodConfigContext, false, SpaceType.INNER_PRODUCT)

        );

        assertTrue(
            validationException.getMessage().contains("Training request ENCODER_PARAMETER_PQ_M is not divisible by vector dimensions")
        );

    }
}
