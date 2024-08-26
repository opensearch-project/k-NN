/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.experimental.UtilityClass;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.quantizationService.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;

@UtilityClass
class QuantizationIndexUtils {

    /**
     * Processes and returns the vector based on whether quantization is applied or not.
     *
     * @param knnVectorValues the KNN vector values to be processed.
     * @param indexBuildSetup the setup containing quantization state and output, along with other parameters.
     * @return the processed vector, either quantized or original.
     * @throws IOException if an I/O error occurs during processing.
     */
    static Object processAndReturnVector(KNNVectorValues<?> knnVectorValues, IndexBuildSetup indexBuildSetup) throws IOException {
        QuantizationService quantizationService = QuantizationService.getInstance();
        if (indexBuildSetup.getQuantizationState() != null && indexBuildSetup.getQuantizationOutput() != null) {
            quantizationService.quantize(
                indexBuildSetup.getQuantizationState(),
                knnVectorValues.getVector(),
                indexBuildSetup.getQuantizationOutput()
            );
            return indexBuildSetup.getQuantizationOutput().getQuantizedVector();
        } else {
            return knnVectorValues.conditionalCloneVector();
        }
    }

    /**
     * Prepares the quantization setup including bytes per vector and dimensions.
     *
     * @param knnVectorValues the KNN vector values.
     * @param indexInfo the index build parameters.
     * @return an instance of QuantizationSetup containing relevant information.
     */
    static IndexBuildSetup prepareIndexBuild(KNNVectorValues<?> knnVectorValues, BuildIndexParams indexInfo) {
        QuantizationState quantizationState = indexInfo.getQuantizationState();
        QuantizationOutput quantizationOutput = null;
        QuantizationService quantizationService = QuantizationService.getInstance();

        int bytesPerVector;
        int dimensions;

        if (quantizationState != null) {
            bytesPerVector = quantizationState.getBytesPerVector();
            dimensions = quantizationState.getDimensions();
            quantizationOutput = quantizationService.createQuantizationOutput(quantizationState.getQuantizationParams());
        } else {
            bytesPerVector = knnVectorValues.bytesPerVector();
            dimensions = knnVectorValues.dimension();
        }

        return new IndexBuildSetup(bytesPerVector, dimensions, quantizationOutput, quantizationState);
    }
}
