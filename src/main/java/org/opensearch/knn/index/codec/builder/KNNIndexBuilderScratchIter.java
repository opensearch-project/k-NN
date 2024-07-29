/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.builder;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.jni.JNIService;

public class KNNIndexBuilderScratchIter extends KNNIndexBuilderScratch {

    @Override
    protected void createIndex() throws IOException {
        long indexAddress = initIndexFromScratch(numDocs, dimension, knnEngine, parameters);
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, getVectorTransfer(vectorDataType), true);
        try {
            for (; !batch.finished; batch = KNNCodecUtil.getVectorBatch(values, getVectorTransfer(vectorDataType), true)) {
                insertToIndex(batch, knnEngine, indexAddress, parameters);
            }
            insertToIndex(batch, knnEngine, indexAddress, parameters);
            writeIndex(indexAddress, indexPath, knnEngine, parameters);
        } catch (Exception e) {
            JNIService.free(indexAddress, knnEngine, VectorDataType.BINARY == vectorDataType);
            if (batch.getVectorAddress() != 0) {
                JNICommons.freeVectorData(batch.getVectorAddress());
            }
            throw e;
        }
    }

    private long initIndexFromScratch(long size, int dim, KNNEngine knnEngine, Map<String, Object> parameters) throws IOException {
        // Pass the path for the nms library to save the file
        return AccessController.doPrivileged((PrivilegedAction<Long>) () -> {
            return JNIService.initIndexFromScratch(size, dim, parameters, knnEngine);
        });
    }

    private void insertToIndex(KNNCodecUtil.VectorBatch batch, KNNEngine knnEngine, long indexAddress, Map<String, Object> parameters)
        throws IOException {
        if (batch.docs.length == 0) {
            return;
        }
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.insertToIndex(batch.docs, batch.getVectorAddress(), batch.getDimension(), parameters, indexAddress, knnEngine);
            return null;
        });
    }

    private void writeIndex(long indexAddress, String indexPath, KNNEngine knnEngine, Map<String, Object> parameters) throws IOException {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.writeIndex(indexPath, indexAddress, knnEngine, parameters);
            return null;
        });
    }
}
