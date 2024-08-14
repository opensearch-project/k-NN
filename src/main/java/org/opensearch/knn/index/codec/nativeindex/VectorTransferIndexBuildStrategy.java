/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapByteVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapFloatVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Transfers all vectors to off heap and then builds an index
 */
final class VectorTransferIndexBuildStrategy implements NativeIndexBuildStrategy {

    private static VectorTransferIndexBuildStrategy INSTANCE = new VectorTransferIndexBuildStrategy();

    public static VectorTransferIndexBuildStrategy getInstance() {
        return INSTANCE;
    }

    private VectorTransferIndexBuildStrategy() {}

    public void buildAndWriteIndex(final BuildIndexParams indexInfo, final KNNVectorValues<?> knnVectorValues) throws IOException {
        // iterating it once to be safe
        knnVectorValues.init();
        try (final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(indexInfo.getVectorDataType())) {
            //Quantization goes here
            List<Integer> tranferredDocIds = new ArrayList<>();
            while (knnVectorValues.docId() != NO_MORE_DOCS) {
                vectorTransfer.transfer(knnVectorValues.getVector());
                tranferredDocIds.add(knnVectorValues.docId());
                knnVectorValues.nextDoc();
            }
            vectorTransfer.flush();

            final Map<String, Object> params = indexInfo.getParameters();
            // Currently this is if else as there are only two cases, with more cases this will have to be made
            // more maintainable
            if (params.containsKey(MODEL_ID)) {
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.createIndexFromTemplate(
                            tranferredDocIds.stream().mapToInt(i -> i).toArray(),
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
                            tranferredDocIds.stream().mapToInt(i -> i).toArray(),
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

    private <T> OffHeapVectorTransfer<T> getVectorTransfer(VectorDataType vectorDataType) throws IOException {
        switch (vectorDataType) {
            case FLOAT:
                return (OffHeapVectorTransfer<T>) new OffHeapFloatVectorTransfer();
            case BINARY:
            case BYTE:
                return (OffHeapVectorTransfer<T>) new OffHeapByteVectorTransfer(100);
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }
}
