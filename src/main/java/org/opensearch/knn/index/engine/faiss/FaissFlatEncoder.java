/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;

/**
 * Flat faiss encoder. Flat encoding means that it does nothing. It needs an encoder, though, because it
 * is used in generating the index description.
 */
public class FaissFlatEncoder implements Encoder {

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(KNNConstants.ENCODER_FLAT)
        .setMapGenerator(
            ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                KNNConstants.FAISS_FLAT_DESCRIPTION,
                methodComponent,
                methodComponentContext
            ).build())
        )
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }
}
