/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;

import java.util.Set;

import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.isFaissSQClipToFP16RangeEnabled;
import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.isFaissSQfp16;

public abstract class AbstractFaissMethod extends AbstractKNNMethod {

    /**
     * Constructor for the AbstractFaissMethod class.
     *
     * @param methodComponent        The method component used to create the method
     * @param spaces                 The set of spaces supported by the method
     * @param knnLibrarySearchContext The KNN library search context
     */
    public AbstractFaissMethod(MethodComponent methodComponent, Set<SpaceType> spaces, KNNLibrarySearchContext knnLibrarySearchContext) {
        super(methodComponent, spaces, knnLibrarySearchContext);
    }

    @Override
    protected PerDimensionValidator doGetPerDimensionValidator(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();
        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
        }

        if (VectorDataType.FLOAT == vectorDataType) {
            if (isFaissSQfp16(knnMethodContext.getMethodComponentContext())) {
                return FaissFP16Util.FP16_VALIDATOR;
            }
            return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
        }

        throw new IllegalStateException("Unsupported vector data type " + vectorDataType);
    }

    @Override
    protected PerDimensionProcessor doGetPerDimensionProcessor(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();

        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        if (VectorDataType.FLOAT == vectorDataType) {
            if (isFaissSQClipToFP16RangeEnabled(knnMethodContext.getMethodComponentContext())) {
                return FaissFP16Util.CLIP_TO_FP16_PROCESSOR;
            }
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }

        throw new IllegalStateException("Unsupported vector data type " + vectorDataType);
    }
}
