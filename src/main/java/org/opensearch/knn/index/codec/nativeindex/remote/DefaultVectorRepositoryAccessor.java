/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.LatchedActionListener;
import org.opensearch.common.CheckedTriFunction;
import org.opensearch.common.StreamContext;
import org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer;
import org.opensearch.common.blobstore.BlobContainer;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.common.blobstore.stream.write.WriteContext;
import org.opensearch.common.blobstore.stream.write.WritePriority;
import org.opensearch.common.io.InputStreamContainer;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.DOC_ID_FILE_EXTENSION;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.VECTORS_PATH;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.VECTOR_BLOB_FILE_EXTENSION;

@Log4j2
@AllArgsConstructor
public class DefaultVectorRepositoryAccessor implements VectorRepositoryAccessor {
    private final BlobStoreRepository repository;
    private final IndexSettings indexSettings;

    /**
     * If the repository implements {@link AsyncMultiStreamBlobContainer}, then parallel uploads will be used. Parallel uploads are backed by a {@link WriteContext}, for which we have a custom
     * {@link org.opensearch.common.blobstore.stream.write.StreamContextSupplier} implementation.
     *
     * @see DefaultVectorRepositoryAccessor#getStreamContext
     * @see DefaultVectorRepositoryAccessor#getTransferPartStreamSupplier
     *
     * @param blobName                  Base name of the blobs we are writing, excluding file extensions
     * @param totalLiveDocs             Number of documents we are processing. This is used to compute the size of the blob we are writing
     * @param vectorDataType            Data type of the vector (FLOAT, BYTE, BINARY)
     * @param knnVectorValuesSupplier   Supplier for {@link KNNVectorValues}
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void writeToRepository(
        String blobName,
        int totalLiveDocs,
        VectorDataType vectorDataType,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier
    ) throws IOException, InterruptedException {
        assert repository != null;
        // Get the blob container based on blobName and the repo base path. This is where the blobs will be written to.
        BlobPath path = repository.basePath().add(indexSettings.getUUID() + VECTORS_PATH);
        BlobContainer blobContainer = repository.blobStore().blobContainer(path);

        KNNVectorValues<?> knnVectorValues = knnVectorValuesSupplier.get();
        initializeVectorValues(knnVectorValues);
        long vectorBlobLength = (long) knnVectorValues.bytesPerVector() * totalLiveDocs;

        if (blobContainer instanceof AsyncMultiStreamBlobContainer) {
            // First initiate vectors upload
            log.debug("Repository {} Supports Parallel Blob Upload", repository);
            // WriteContext is the main entry point into asyncBlobUpload. It stores all of our upload configurations, analogous to
            // BuildIndexParams
            WriteContext writeContext = new WriteContext.Builder().fileName(blobName + VECTOR_BLOB_FILE_EXTENSION)
                .streamContextSupplier((partSize) -> getStreamContext(partSize, vectorBlobLength, knnVectorValuesSupplier, vectorDataType))
                .fileSize(vectorBlobLength)
                .failIfAlreadyExists(true)
                .writePriority(WritePriority.NORMAL)
                // TODO: Checksum implementations -- It is difficult to calculate a checksum on the knnVectorValues as
                // there is no underlying file upon which we can create the checksum. We should be able to create a
                // checksum still by iterating through once, however this will be an expensive operation.
                .uploadFinalizer((bool) -> {})
                .doRemoteDataIntegrityCheck(false)
                .expectedChecksum(null)
                .build();

            AtomicReference<Exception> exception = new AtomicReference<>();
            final CountDownLatch latch = new CountDownLatch(1);
            ((AsyncMultiStreamBlobContainer) blobContainer).asyncBlobUpload(
                writeContext,
                new LatchedActionListener<>(new ActionListener<>() {
                    @Override
                    public void onResponse(Void unused) {
                        log.debug(
                            "Parallel vector upload succeeded for blob {} with size {}",
                            blobName + VECTOR_BLOB_FILE_EXTENSION,
                            vectorBlobLength
                        );
                    }

                    @Override
                    public void onFailure(Exception e) {
                        log.error(
                            "Parallel vector upload failed for blob {} with size {}",
                            blobName + VECTOR_BLOB_FILE_EXTENSION,
                            vectorBlobLength,
                            e
                        );
                        exception.set(e);
                    }
                }, latch)
            );

            // Then upload doc id blob before waiting on vector uploads
            // TODO: We wrap with a BufferedInputStream to support retries. We can tune this buffer size to optimize performance.
            // Note: We do not use the parallel upload API here as the doc id blob will be much smaller than the vector blob
            writeDocIds(knnVectorValuesSupplier.get(), vectorBlobLength, totalLiveDocs, blobName, blobContainer);
            latch.await();
            if (exception.get() != null) {
                throw new IOException(exception.get());
            }
        } else {
            log.debug("Repository {} Does Not Support Parallel Blob Upload", repository);
            // Write Vectors
            InputStream vectorStream = new BufferedInputStream(new VectorValuesInputStream(knnVectorValuesSupplier.get(), vectorDataType));
            log.debug("Writing {} bytes for {} docs to {}", vectorBlobLength, totalLiveDocs, blobName + VECTOR_BLOB_FILE_EXTENSION);
            blobContainer.writeBlob(blobName + VECTOR_BLOB_FILE_EXTENSION, vectorStream, vectorBlobLength, true);
            // Then write doc ids
            writeDocIds(knnVectorValuesSupplier.get(), vectorBlobLength, totalLiveDocs, blobName, blobContainer);
        }
    }

    /**
     * Helper method for uploading doc ids to repository, as it's re-used in both parallel and sequential upload cases
     * @param knnVectorValues
     * @param vectorBlobLength
     * @param totalLiveDocs
     * @param blobName
     * @param blobContainer
     * @throws IOException
     */
    private void writeDocIds(
        KNNVectorValues<?> knnVectorValues,
        long vectorBlobLength,
        long totalLiveDocs,
        String blobName,
        BlobContainer blobContainer
    ) throws IOException {
        InputStream docStream = new BufferedInputStream(new DocIdInputStream(knnVectorValues));
        log.debug(
            "Writing {} bytes for {} docs ids to {}",
            vectorBlobLength,
            totalLiveDocs * Integer.BYTES,
            blobName + DOC_ID_FILE_EXTENSION
        );
        blobContainer.writeBlob(blobName + DOC_ID_FILE_EXTENSION, docStream, totalLiveDocs * Integer.BYTES, true);
    }

    /**
     * Returns a {@link org.opensearch.common.StreamContext}. Intended to be invoked as a {@link org.opensearch.common.blobstore.stream.write.StreamContextSupplier},
     * which takes the partSize determined by the repository implementation and calculates the number of parts as well as handles the last part of the stream.
     *
     * @see DefaultVectorRepositoryAccessor#getTransferPartStreamSupplier
     *
     * @param partSize                  Size of each InputStream to be uploaded in parallel. Provided by repository implementation
     * @param vectorBlobLength          Total size of the vectors across all InputStreams
     * @param knnVectorValuesSupplier   Supplier for {@link KNNVectorValues}
     * @param vectorDataType            Data type of the vector (FLOAT, BYTE, BINARY)
     * @return a {@link org.opensearch.common.StreamContext} with a function that will create {@link InputStream}s of {@param partSize}
     */
    private StreamContext getStreamContext(
        long partSize,
        long vectorBlobLength,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        VectorDataType vectorDataType
    ) {
        long lastPartSize = (vectorBlobLength % partSize) != 0 ? vectorBlobLength % partSize : partSize;
        int numberOfParts = (int) ((vectorBlobLength % partSize) == 0 ? vectorBlobLength / partSize : (vectorBlobLength / partSize) + 1);
        return new StreamContext(
            getTransferPartStreamSupplier(knnVectorValuesSupplier, vectorDataType),
            partSize,
            lastPartSize,
            numberOfParts
        );
    }

    /**
     * This method handles creating {@link VectorValuesInputStream}s based on the part number, the requested size of the stream part, and the position that the stream starts at within the underlying {@link KNNVectorValues}
     *
     * @param knnVectorValuesSupplier       Supplier for {@link KNNVectorValues}
     * @param vectorDataType                Data type of the vector (FLOAT, BYTE, BINARY)
     * @return a function with which the repository implementation will use to create {@link VectorValuesInputStream}s of specific sizes and start positions.
     */
    private CheckedTriFunction<Integer, Long, Long, InputStreamContainer, IOException> getTransferPartStreamSupplier(
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        VectorDataType vectorDataType
    ) {
        return ((partNo, size, position) -> {
            log.info("Creating InputStream for partNo: {}, size: {}, position: {}", partNo, size, position);
            VectorValuesInputStream vectorValuesInputStream = new VectorValuesInputStream(
                knnVectorValuesSupplier.get(),
                vectorDataType,
                position,
                size
            );
            return new InputStreamContainer(vectorValuesInputStream, size, position);
        });
    }

    @Override
    public void readFromRepository(String path, IndexOutputWithBuffer indexOutputWithBuffer) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new IllegalArgumentException("download path is null or empty");
        }
        Path downloadPath = Paths.get(path);
        String fileName = downloadPath.getFileName().toString();
        if (!fileName.endsWith(KNNEngine.FAISS.getExtension())) {
            log.error("download path [{}] does not end with extension [{}}", downloadPath, KNNEngine.FAISS.getExtension());
            throw new IllegalArgumentException("download path has incorrect file extension");
        }

        BlobPath blobContainerPath = new BlobPath();
        if (downloadPath.getParent() != null) {
            for (Path p : downloadPath.getParent()) {
                blobContainerPath = blobContainerPath.add(p.getFileName().toString());
            }
        }

        BlobContainer blobContainer = repository.blobStore().blobContainer(blobContainerPath);
        // TODO: We are using the sequential download API as multi-part parallel download is difficult for us to implement today and
        // requires some changes in core. For more details, see: https://github.com/opensearch-project/k-NN/issues/2464
        InputStream graphStream = blobContainer.readBlob(fileName);
        indexOutputWithBuffer.writeFromStreamWithBuffer(graphStream);
    }
}
