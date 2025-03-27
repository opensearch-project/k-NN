/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.junit.Before;
import org.mockito.Mockito;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.common.SetOnce;
import org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.common.blobstore.DeleteResult;
import org.opensearch.common.blobstore.fs.FsBlobContainer;
import org.opensearch.common.blobstore.fs.FsBlobStore;
import org.opensearch.common.blobstore.stream.read.ReadContext;
import org.opensearch.common.blobstore.stream.write.WriteContext;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.BUCKET;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.S3;

/**
 * Base test class for remote index build tests
 */
abstract class RemoteIndexBuildTests extends KNNTestCase {

    public static final String TEST_BUCKET = "test-bucket";
    public static final String TEST_CLUSTER = "test-cluster";
    final List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 }, new float[] { 3, 4 });
    final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(vectorValues);
    final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = KNNVectorValuesFactory.getVectorValuesSupplier(
        VectorDataType.FLOAT,
        randomVectorValues
    );
    final IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
    final String segmentName = "test-segment-name";
    final SegmentInfo segmentInfo = new SegmentInfo(
        mock(Directory.class),
        mock(Version.class),
        mock(Version.class),
        segmentName,
        0,
        false,
        false,
        mock(Codec.class),
        mock(Map.class),
        new byte[16],
        mock(Map.class),
        mock(Sort.class)
    );
    final SegmentWriteState segmentWriteState = new SegmentWriteState(
        mock(InfoStream.class),
        mock(Directory.class),
        segmentInfo,
        mock(FieldInfos.class),
        null,
        mock(IOContext.class)
    );
    final KNNVectorValues<?> knnVectorValues = knnVectorValuesSupplier.get();
    final BuildIndexParams buildIndexParams = BuildIndexParams.builder()
        .indexOutputWithBuffer(indexOutputWithBuffer)
        .knnEngine(KNNEngine.FAISS)
        .vectorDataType(VectorDataType.FLOAT)
        .parameters(Map.of("index", "param"))
        .knnVectorValuesSupplier(knnVectorValuesSupplier)
        .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
        .segmentWriteState(segmentWriteState)
        .build();

    record TestIndexBuildStrategy(SetOnce<Boolean> fallback) implements NativeIndexBuildStrategy {
        @Override
        public void buildAndWriteIndex(BuildIndexParams indexInfo) throws IOException {
            fallback.set(true);
        }
    }

    static class TestAsyncBlobContainer extends FsBlobContainer implements AsyncMultiStreamBlobContainer {
        private final boolean throwsException;

        public TestAsyncBlobContainer(FsBlobStore blobStore, BlobPath blobPath, Path path, boolean throwsException) {
            super(blobStore, blobPath, path);
            this.throwsException = throwsException;

        }

        @Override
        public void asyncBlobUpload(WriteContext writeContext, ActionListener<Void> actionListener) throws IOException {
            if (this.throwsException) {
                actionListener.onFailure(new IOException("Test Exception"));
            } else {
                actionListener.onResponse(null);
            }
        }

        @Override
        public void writeBlob(String blobName, InputStream inputStream, long blobSize, boolean failIfAlreadyExists) throws IOException {}

        @Override
        public void readBlobAsync(String s, ActionListener<ReadContext> actionListener) {}

        @Override
        public boolean remoteIntegrityCheckSupported() {
            return true;
        }

        @Override
        public void deleteAsync(ActionListener<DeleteResult> actionListener) {}

        @Override
        public void deleteBlobsAsyncIgnoringIfNotExists(List<String> list, ActionListener<Void> actionListener) {}
    }

    static class TestBlobContainer extends FsBlobContainer {

        public TestBlobContainer(FsBlobStore blobStore, BlobPath blobPath, Path path) {
            super(blobStore, blobPath, path);
        }

        @Override
        public void writeBlob(String blobName, InputStream inputStream, long blobSize, boolean failIfAlreadyExists) {}
    }

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();
        ClusterSettings clusterSettings = mock(ClusterSettings.class);
        when(clusterSettings.get(KNN_REMOTE_VECTOR_REPO_SETTING)).thenReturn("test-repo-name");
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
    }

    public static IndexSettings createTestIndexSettings() {
        IndexSettings mockIndexSettings = mock(IndexSettings.class);
        Settings indexSettingsSettings = Settings.builder().put(ClusterName.CLUSTER_NAME_SETTING.getKey(), TEST_CLUSTER).build();
        when(mockIndexSettings.getSettings()).thenReturn(indexSettingsSettings);
        return mockIndexSettings;
    }

    public static RepositoryMetadata createTestRepositoryMetadata() {
        RepositoryMetadata metadata = mock(RepositoryMetadata.class);
        Settings repoSettings = Settings.builder().put(BUCKET, TEST_BUCKET).build();
        when(metadata.type()).thenReturn(S3);
        when(metadata.settings()).thenReturn(repoSettings);
        return metadata;
    }
}
