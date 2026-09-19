/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Set;

/**
 * Flat faiss encoder. Flat encoding means that it does nothing. It needs an encoder, though, because it
 * is used in generating the index description.
 */
public class FaissFlatEncoder implements Encoder {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(
        VectorDataType.FLOAT,
        VectorDataType.BYTE,
        VectorDataType.BINARY,
        VectorDataType.HALF_FLOAT
    );

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(KNNConstants.ENCODER_FLAT)
        .setKnnLibraryIndexingContextGenerator(
            ((methodComponent, methodComponentContext, knnMethodConfigContext) -> MethodAsMapBuilder.builder(
                indexDescriptionFor(knnMethodConfigContext.getVectorDataType()),
                methodComponent,
                methodComponentContext,
                knnMethodConfigContext
            ).build())
        )
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .build();

    /**
     * half_float's native type is already fp16, so "flat" (no quantization) builds an
     * {@code IndexScalarQuantizer(QT_fp16)} instead of {@code IndexFlat}: fp32 -> fp16 is a lossless cast.
     */
    private static String indexDescriptionFor(VectorDataType vectorDataType) {
        return vectorDataType == VectorDataType.HALF_FLOAT
            ? KNNConstants.FAISS_SQ_DESCRIPTION + KNNConstants.FAISS_SQ_ENCODER_FP16
            : KNNConstants.FAISS_FLAT_DESCRIPTION;
    }

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext encoderContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        return CompressionLevel.x1;
    }
}
