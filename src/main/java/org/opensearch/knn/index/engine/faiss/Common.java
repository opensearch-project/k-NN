/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Map;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_TYPES;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;

// TODO: Remove class once encoders are refactored
class Common {
    final static Map<String, MethodComponent> COMMON_ENCODERS = ImmutableMap.of(
        KNNConstants.ENCODER_FLAT,
        MethodComponent.Builder.builder(KNNConstants.ENCODER_FLAT)
            .setMapGenerator(
                ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                    KNNConstants.FAISS_FLAT_DESCRIPTION,
                    methodComponent,
                    methodComponentContext
                ).build())
            )
            .build(),
        ENCODER_SQ,
        MethodComponent.Builder.builder(ENCODER_SQ)
            .addParameter(
                FAISS_SQ_TYPE,
                new Parameter.StringParameter(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16, FAISS_SQ_ENCODER_TYPES::contains)
            )
            .addParameter(FAISS_SQ_CLIP, new Parameter.BooleanParameter(FAISS_SQ_CLIP, false, Objects::nonNull))
            .setMapGenerator(
                ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                    FAISS_SQ_DESCRIPTION,
                    methodComponent,
                    methodComponentContext
                ).addParameter(FAISS_SQ_TYPE, "", "").build())
            )
            .build()
    );
}
