/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.mockito.Mockito;
import org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer;
import org.opensearch.common.blobstore.BlobContainer;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.common.blobstore.BlobStore;
import org.opensearch.common.blobstore.fs.FsBlobStore;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;
import java.nio.file.Path;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy.DOC_ID_FILE_EXTENSION;
import static org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy.VECTOR_BLOB_FILE_EXTENSION;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

public class DefaultVectorRepositoryAccessorTests extends RemoteIndexBuildTests {

    /**
     * Test that whenever an non AsyncMultiStreamBlobContainer is used, writeBlob is invoked twice
     */
    public void testRepositoryInteractionWithAsyncMultiStreamBlobContainer() throws IOException, InterruptedException {
        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        BlobStoreRepository mockRepository = mock(BlobStoreRepository.class);
        BlobPath testBasePath = new BlobPath().add("testBasePath");
        BlobStore mockBlobStore = mock(BlobStore.class);

        when(repositoriesService.repository(any())).thenReturn(mockRepository);
        when(mockRepository.basePath()).thenReturn(testBasePath);
        when(mockRepository.blobStore()).thenReturn(mockBlobStore);

        BlobContainer testContainer = Mockito.spy(new TestBlobContainer(mock(FsBlobStore.class), testBasePath, mock(Path.class)));
        when(mockBlobStore.blobContainer(any())).thenReturn(testContainer);

        VectorRepositoryAccessor objectUnderTest = new DefaultVectorRepositoryAccessor(mockRepository, mock(IndexSettings.class));

        String BLOB_NAME = "test_blob";
        int NUM_DOCS = 100;
        objectUnderTest.writeToRepository(BLOB_NAME, NUM_DOCS, VectorDataType.FLOAT, knnVectorValuesSupplier);

        KNNVectorValues<?> knnVectorValues = knnVectorValuesSupplier.get();
        initializeVectorValues(knnVectorValues);
        verify(testContainer).writeBlob(
            eq(BLOB_NAME + VECTOR_BLOB_FILE_EXTENSION),
            any(),
            eq((long) NUM_DOCS * knnVectorValues.bytesPerVector()),
            eq(true)
        );
        verify(testContainer).writeBlob(eq(BLOB_NAME + DOC_ID_FILE_EXTENSION), any(), eq((long) NUM_DOCS * Integer.BYTES), eq(true));
        verify(mockBlobStore).blobContainer(any());
        verify(mockRepository).basePath();
    }

    /**
     * Test that whenever an AsyncMultiStreamBlobContainer is used, both asyncBlobUpload and writeBlob are invoked once and only once
     */
    public void testRepositoryInteractionWithBlobContainer() throws IOException, InterruptedException {
        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        BlobStoreRepository mockRepository = mock(BlobStoreRepository.class);
        BlobPath testBasePath = new BlobPath().add("testBasePath");
        BlobStore mockBlobStore = mock(BlobStore.class);

        when(repositoriesService.repository(any())).thenReturn(mockRepository);
        when(mockRepository.basePath()).thenReturn(testBasePath);
        when(mockRepository.blobStore()).thenReturn(mockBlobStore);

        AsyncMultiStreamBlobContainer testContainer = Mockito.spy(
            new TestAsyncBlobContainer(mock(FsBlobStore.class), testBasePath, mock(Path.class), false)
        );
        when(mockBlobStore.blobContainer(any())).thenReturn(testContainer);

        VectorRepositoryAccessor objectUnderTest = new DefaultVectorRepositoryAccessor(mockRepository, mock(IndexSettings.class));

        String BLOB_NAME = "test_blob";
        int NUM_DOCS = 100;
        objectUnderTest.writeToRepository(BLOB_NAME, NUM_DOCS, VectorDataType.FLOAT, knnVectorValuesSupplier);

        verify(testContainer).asyncBlobUpload(any(), any());
        verify(testContainer).writeBlob(eq(BLOB_NAME + DOC_ID_FILE_EXTENSION), any(), eq((long) NUM_DOCS * Integer.BYTES), eq(true));
        verify(mockBlobStore).blobContainer(any());
        verify(mockRepository).basePath();
    }

    /**
     * Test that when an exception is thrown during asyncBlobUpload, the exception is rethrown.
     */
    public void testAsyncUploadThrowsException() throws InterruptedException, IOException {
        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        BlobStoreRepository mockRepository = mock(BlobStoreRepository.class);
        BlobPath testBasePath = new BlobPath().add("testBasePath");
        BlobStore mockBlobStore = mock(BlobStore.class);

        when(repositoriesService.repository(any())).thenReturn(mockRepository);
        when(mockRepository.basePath()).thenReturn(testBasePath);
        when(mockRepository.blobStore()).thenReturn(mockBlobStore);

        AsyncMultiStreamBlobContainer testContainer = Mockito.spy(
            new TestAsyncBlobContainer(mock(FsBlobStore.class), testBasePath, mock(Path.class), true)
        );
        when(mockBlobStore.blobContainer(any())).thenReturn(testContainer);

        VectorRepositoryAccessor objectUnderTest = new DefaultVectorRepositoryAccessor(mockRepository, mock(IndexSettings.class));

        String BLOB_NAME = "test_blob";
        int NUM_DOCS = 100;
        assertThrows(
            IOException.class,
            () -> objectUnderTest.writeToRepository(BLOB_NAME, NUM_DOCS, VectorDataType.FLOAT, knnVectorValuesSupplier)
        );

        verify(testContainer).asyncBlobUpload(any(), any());
        // Doc ids should still get written because exception is handled after awaiting on asyncBlobUpload
        verify(testContainer).writeBlob(eq(BLOB_NAME + DOC_ID_FILE_EXTENSION), any(), eq((long) NUM_DOCS * Integer.BYTES), eq(true));
        verify(mockBlobStore).blobContainer(any());
        verify(mockRepository).basePath();
    }
}
