/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.function.Supplier;

/**
 * Interface which dictates how we use we interact with a {@link org.opensearch.repositories.blobstore.BlobStoreRepository} from {@link RemoteIndexBuildStrategy}
 */
interface VectorRepositoryAccessor {
    /**
     * This method is responsible for writing both the vector blobs and doc ids provided by {@param knnVectorValuesSupplier} to the configured repository
     *
     * @param blobName                  Base name of the blobs we are writing, excluding file extensions
     * @param totalLiveDocs             Number of documents we are processing. This is used to compute the size of the blob we are writing
     * @param vectorDataType            Data type of the vector (FLOAT, BYTE, BINARY)
     * @param knnVectorValuesSupplier   Supplier for {@link org.opensearch.knn.index.vectorvalues.KNNVectorValues}
     * @throws java.io.IOException
     * @throws InterruptedException
     */
    void writeToRepository(
        String blobName,
        int totalLiveDocs,
        VectorDataType vectorDataType,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier
    ) throws IOException, InterruptedException;

    /**
     * Read constructed vector file from remote repository and write to IndexOutput
     * @param fileName                      File name as String
     * @param indexOutputWithBuffer     {@link IndexOutputWithBuffer} which will be used to write to the underlying {@link org.apache.lucene.store.IndexOutput}
     * @throws IOException
     */
    void readFromRepository(String fileName, IndexOutputWithBuffer indexOutputWithBuffer) throws IOException;
}
