/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.mockito.Mockito;
import org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer;
import org.opensearch.common.blobstore.BlobContainer;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.common.blobstore.BlobStore;
import org.opensearch.common.blobstore.fs.FsBlobStore;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

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

    /**
     * Verify the buffered read method in {@link DefaultVectorRepositoryAccessor#readFromRepository} produces the correct result
     */
    public void testRepositoryRead() throws IOException {
        String TEST_FILE_NAME = randomAlphaOfLength(8) + KNNEngine.FAISS.getExtension();

        // Create an InputStream with random values
        int TEST_ARRAY_SIZE = 64 * 1024 * 10;
        byte[] byteArray = new byte[TEST_ARRAY_SIZE];
        Random random = new Random();
        random.nextBytes(byteArray);
        InputStream randomStream = new ByteArrayInputStream(byteArray);

        // Create a test segment that we will read/write from
        Directory directory;
        directory = newFSDirectory(createTempDir());
        String TEST_SEGMENT_NAME = "test-segment-name";
        IndexOutput testIndexOutput = directory.createOutput(TEST_SEGMENT_NAME, IOContext.DEFAULT);
        IndexOutputWithBuffer testIndexOutputWithBuffer = new IndexOutputWithBuffer(testIndexOutput);

        // Set up RemoteIndexBuildStrategy and write to IndexOutput
        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        BlobStoreRepository mockRepository = mock(BlobStoreRepository.class);
        BlobPath testBasePath = new BlobPath().add("testBasePath");
        BlobStore mockBlobStore = mock(BlobStore.class);
        AsyncMultiStreamBlobContainer mockBlobContainer = mock(AsyncMultiStreamBlobContainer.class);

        when(repositoriesService.repository(any())).thenReturn(mockRepository);
        when(mockRepository.basePath()).thenReturn(testBasePath);
        when(mockRepository.blobStore()).thenReturn(mockBlobStore);
        when(mockBlobStore.blobContainer(any())).thenReturn(mockBlobContainer);
        when(mockBlobContainer.readBlob(TEST_FILE_NAME)).thenReturn(randomStream);

        VectorRepositoryAccessor objectUnderTest = new DefaultVectorRepositoryAccessor(mockRepository, mock(IndexSettings.class));

        // Verify file extension check
        assertThrows(IllegalArgumentException.class, () -> objectUnderTest.readFromRepository("test_file.txt", testIndexOutputWithBuffer));

        // Now test with valid file extensions
        String testPath = randomFrom(
            List.of(
                "testBasePath/testDirectory/" + TEST_FILE_NAME, // Test with subdirectory
                "testBasePath/" + TEST_FILE_NAME, // Test with only base path
                TEST_FILE_NAME // test with no base path
            )
        );
        // This should read from randomStream into testIndexOutput
        objectUnderTest.readFromRepository(testPath, testIndexOutputWithBuffer);
        testIndexOutput.close();

        // Now try to read from the IndexOutput
        IndexInput testIndexInput = directory.openInput(TEST_SEGMENT_NAME, IOContext.DEFAULT);
        byte[] resultByteArray = new byte[TEST_ARRAY_SIZE];
        testIndexInput.readBytes(resultByteArray, 0, TEST_ARRAY_SIZE);
        assertArrayEquals(byteArray, resultByteArray);

        // Test Cleanup
        testIndexInput.close();
        directory.close();
    }
}
