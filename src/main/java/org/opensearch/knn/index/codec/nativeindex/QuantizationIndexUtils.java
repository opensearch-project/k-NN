/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.experimental.UtilityClass;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;

@UtilityClass
public class QuantizationIndexUtils {

    /**
     * Processes the vector from {@link KNNVectorValues} and returns either a cloned quantized vector or a cloned original vector.
     *
     * @param knnVectorValues The KNN vector values containing the original vector.
     * @param indexBuildSetup The setup containing the quantization state and output details.
     * @return The quantized vector (as a byte array) or the original/cloned vector.
     * @throws IOException If an I/O error occurs while processing the vector.
     */
    public static Object processAndReturnVector(KNNVectorValues<?> knnVectorValues, IndexBuildSetup indexBuildSetup) throws IOException {
        QuantizationService quantizationService = QuantizationService.getInstance();
        if (indexBuildSetup.getQuantizationState() != null && indexBuildSetup.getQuantizationOutput() != null) {
            quantizationService.quantize(
                indexBuildSetup.getQuantizationState(),
                knnVectorValues.getVector(),
                indexBuildSetup.getQuantizationOutput()
            );
            /**
             * Returns a copy of the quantized vector. This is because of during transfer same vectors was getting
             * added due to reference.
             */
            return indexBuildSetup.getQuantizationOutput().getQuantizedVectorCopy();
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
    public static IndexBuildSetup prepareIndexBuild(KNNVectorValues<?> knnVectorValues, BuildIndexParams indexInfo) {
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
