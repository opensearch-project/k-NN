/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNVectorUtil.intListToArray;
import static org.opensearch.knn.common.KNNVectorUtil.iterateVectorValuesOnce;
import static org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory.getVectorTransfer;

/**
 * Transfers all vectors to off heap and then builds an index
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
final class DefaultIndexBuildStrategy implements NativeIndexBuildStrategy {

    private static DefaultIndexBuildStrategy INSTANCE = new DefaultIndexBuildStrategy();

    public static DefaultIndexBuildStrategy getInstance() {
        return INSTANCE;
    }

    public void buildAndWriteIndex(final BuildIndexParams indexInfo, final KNNVectorValues<?> knnVectorValues) throws IOException {
        iterateVectorValuesOnce(knnVectorValues); // to get bytesPerVector
        int transferLimit = (int) Math.max(1, KNNSettings.getVectorStreamingMemoryLimit().getBytes() / knnVectorValues.bytesPerVector());
        try (final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(indexInfo.getVectorDataType(), transferLimit)) {

            final List<Integer> tranferredDocIds = new ArrayList<>();
            while (knnVectorValues.docId() != NO_MORE_DOCS) {
                // append is true here so off heap memory buffer isn't overwritten
                vectorTransfer.transfer(knnVectorValues.conditionalCloneVector(), true);
                tranferredDocIds.add(knnVectorValues.docId());
                knnVectorValues.nextDoc();
            }
            vectorTransfer.flush(true);

            final Map<String, Object> params = indexInfo.getParameters();
            long vectorAddress = vectorTransfer.getVectorAddress();
            // Currently this is if else as there are only two cases, with more cases this will have to be made
            // more maintainable
            if (params.containsKey(MODEL_ID)) {
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.createIndexFromTemplate(
                        intListToArray(tranferredDocIds),
                        vectorAddress,
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
                        intListToArray(tranferredDocIds),
                        vectorAddress,
                        knnVectorValues.dimension(),
                        indexInfo.getIndexPath(),
                        indexInfo.getParameters(),
                        indexInfo.getKnnEngine()
                    );
                    return null;
                });
            }
            // Resetting here as vectors are deleted in JNILayer for non-iterative index builds
            vectorTransfer.reset();
        } catch (Exception exception) {
            throw new RuntimeException(
                "Failed to build index, field name " + indexInfo.getFieldName() + ", parameters " + indexInfo,
                exception
            );
        }
    }
}
