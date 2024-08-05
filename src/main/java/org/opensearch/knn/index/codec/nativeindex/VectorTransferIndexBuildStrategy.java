/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapByteQuantizedVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapFloatVectorTransfer;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Transfers all vectors to offheap and then builds an index
 */
final class VectorTransferIndexBuildStrategy implements NativeIndexBuildStrategy {

    private static VectorTransferIndexBuildStrategy INSTANCE = new VectorTransferIndexBuildStrategy();

    public static VectorTransferIndexBuildStrategy getInstance() {
        return INSTANCE;
    }

    private VectorTransferIndexBuildStrategy() {}

    public void buildAndWriteIndex(final BuildIndexParams indexInfo, final KNNVectorValues<?> knnVectorValues) throws IOException {
        //iterating it once to be safe
        knnVectorValues.init();
        try (final VectorTransfer vectorTransfer = getVectorTransfer(indexInfo.getVectorDataType(), knnVectorValues)) {
            vectorTransfer.transferBatch();
            assert !vectorTransfer.hasNext();

            final Map<String, Object> params = indexInfo.getParameters();
            // Currently this is if else as there are only two cases, with more cases this will have to be made
            // more maintainable
            if (params.containsKey(MODEL_ID)) {
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.createIndexFromTemplate(
                        vectorTransfer.getTransferredDocsIds(),
                        vectorTransfer.getVectorAddress(),
                        knnVectorValues.dimension(),
                        indexInfo.getIndexPath(),
                        (byte[]) params.get(KNNConstants.MODEL_BLOB_PARAMETER),
                        indexInfo.getParameters(),
                        indexInfo.getKnnEngine()
                    );
                    return null;
                });
            } else {
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.createIndex(
                        vectorTransfer.getTransferredDocsIds(),
                        vectorTransfer.getVectorAddress(),
                        knnVectorValues.dimension(),
                        indexInfo.getIndexPath(),
                        indexInfo.getParameters(),
                        indexInfo.getKnnEngine()
                    );
                    return null;
                });
            }

        } catch (Exception exception) {
            throw new RuntimeException("Failed to build index", exception);
        }
    }

    private VectorTransfer getVectorTransfer(VectorDataType vectorDataType, KNNVectorValues<?> knnVectorValues) throws IOException {
        switch (vectorDataType) {
            case FLOAT:
                return new OffHeapFloatVectorTransfer((KNNFloatVectorValues) knnVectorValues, knnVectorValues.totalLiveDocs());
            case BINARY:
            case BYTE:
                return new OffHeapByteQuantizedVectorTransfer<>((KNNVectorValues<byte[]>) knnVectorValues, knnVectorValues.totalLiveDocs());
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }
}
