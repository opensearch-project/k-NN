/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Package which includes the classes used for building vector indices remotely.
 * <p>
 *     For repository uploads, there are 2 methods we can use -- [1] {@link org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer#asyncBlobUpload}, which uses multiple {@link java.io.InputStream}s
 *     to perform blob upload in parallel across streams and [2] {@link org.opensearch.common.blobstore.BlobContainer#writeBlob}, which uses a single {@link java.io.InputStream} to perform blob upload sequentially.
 * </p>
 * <p>
 *     In order to optimize the performance of vector blob uploads, we use the parallel upload method to upload vector blobs, as since doc id blobs will be relatively small we only use the sequential upload method there.
 *     The parallel blob upload method takes in {@link org.opensearch.common.blobstore.stream.write.WriteContext}, which takes in a {@link org.opensearch.common.blobstore.stream.write.StreamContextSupplier}
 *     that subsequently creates {@link org.opensearch.common.io.InputStreamContainer}s of a part size determined by the repository.
 * </p>
 * <p>
 *     We are splitting the {@link org.opensearch.knn.index.vectorvalues.KNNVectorValues} into N streams, however since it is an iterator we will need to create N instances in order to iterator through the vector
 *     values in parallel. {@link org.opensearch.knn.index.codec.nativeindex.remote.VectorValuesInputStream} takes in both a position and a size in the constructor arguments, which will iterate to the exact
 *     byte specified by position and set the head of the InputStream to that position. The stream will then only allow size bytes to be read from it.
 * </p>
 * <p>
 *     The part size (and therefore number of parts) is determined by the repository implementation, so from this package we are only responsible for creating correctly sized and positioned InputStreams based on the
 *     the part size requested by the repository.
 * </p>
 */
package org.opensearch.knn.index.codec.nativeindex.remote;
