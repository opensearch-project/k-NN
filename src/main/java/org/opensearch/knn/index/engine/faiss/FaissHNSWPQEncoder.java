/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.FAISS_PQ_DESCRIPTION;

/**
 * Faiss HNSW PQ encoder. Right now, the implementations are slightly different during validation between this an
 * {@link FaissIVFPQEncoder}. Hence, they are separate classes.
 */
public class FaissHNSWPQEncoder implements Encoder {

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(KNNConstants.ENCODER_PQ)
        .addParameter(
            ENCODER_PARAMETER_PQ_M,
            new Parameter.IntegerParameter(
                ENCODER_PARAMETER_PQ_M,
                ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT,
                v -> v > 0 && v < ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT,
                (v, vectorSpaceInfo) -> vectorSpaceInfo.getDimension() % v == 0
            )
        )
        .addParameter(
            ENCODER_PARAMETER_PQ_CODE_SIZE,
            new Parameter.IntegerParameter(
                ENCODER_PARAMETER_PQ_CODE_SIZE,
                ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT,
                v -> Objects.equals(v, ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT)
            )
        )
        .setRequiresTraining(true)
        .setMapGenerator(
            ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                FAISS_PQ_DESCRIPTION,
                methodComponent,
                methodComponentContext
            ).addParameter(ENCODER_PARAMETER_PQ_M, "", "").build())
        )
        .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
            int codeSize = ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
            return ((4L * (1L << codeSize) * dimension) / BYTES_PER_KILOBYTES) + 1;
        })
        .build();

    @Override
    public String getName() {
        return METHOD_COMPONENT.getName();
    }

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }
}
