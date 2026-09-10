/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ.grpc;

import com.google.protobuf.ByteString;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.protobufs.BulkRequest;
import org.opensearch.protobufs.BulkRequestBody;
import org.opensearch.protobufs.BulkResponse;
import org.opensearch.protobufs.IndexOperation;
import org.opensearch.protobufs.OperationContainer;
import org.opensearch.protobufs.Refresh;
import org.opensearch.protobufs.services.DocumentServiceGrpc;

import java.io.ByteArrayOutputStream;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;

import static io.grpc.internal.GrpcUtil.NOOP_PROXY_DETECTOR;

/**
 * Base class for k-NN gRPC transport integration tests.
 *
 * <p>These run against the {@code integTest} REST test cluster, which binds the gRPC aux transport
 * ({@code transport-grpc} module, bundled in the ARCHIVE distribution) on a fixed port configured in
 * build.gradle. The test builds a plaintext gRPC {@link ManagedChannel} to {@code 127.0.0.1:<port>} and
 * issues {@link DocumentServiceGrpc} calls. The test code intentionally references only the public
 * protobuf stubs + a gRPC client — never the transport-grpc module's server classes — because that module
 * is not a resolvable test dependency (only {@code transport-grpc-spi} and {@code protobufs} are).
 *
 * <p>The gRPC port is passed from the build via the {@code tests.grpc.port} system property and must match
 * the {@code aux.transport.transport-grpc.port} cluster setting.
 */
public abstract class KNNGrpcIntegTestCase extends KNNRestTestCase {

    protected int grpcPort() {
        return Integer.parseInt(System.getProperty("tests.grpc.port", "9400"));
    }

    /**
     * Opens a plaintext gRPC channel to the test cluster's gRPC aux transport. Caller is responsible for
     * shutting it down (see {@link #shutdownChannel(ManagedChannel)}).
     */
    protected ManagedChannel newGrpcChannel() {
        return NettyChannelBuilder.forAddress("127.0.0.1", grpcPort()).proxyDetector(NOOP_PROXY_DETECTOR).usePlaintext().build();
    }

    protected void shutdownChannel(ManagedChannel channel) throws InterruptedException {
        if (channel == null) {
            return;
        }
        channel.shutdown();
        if (!channel.awaitTermination(5, TimeUnit.SECONDS)) {
            channel.shutdownNow();
            channel.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    /**
     * Encodes a document map into the given XContent media type (JSON / CBOR / SMILE) as raw bytes, exactly
     * as a gRPC client would place them in {@link BulkRequestBody}'s {@code object} field.
     */
    protected ByteString encodeSource(Map<String, Object> source, MediaType mediaType) throws Exception {
        try (ByteArrayOutputStream os = new ByteArrayOutputStream()) {
            XContentBuilder builder = MediaTypeRegistry.contentBuilder(mediaType, os).map(source);
            builder.close();
            return ByteString.copyFrom(os.toByteArray());
        }
    }

    /**
     * Indexes a single document over gRPC into {@code index} with {@code id}, encoding the source in
     * {@code mediaType}. Uses {@code refresh=true} so the doc is immediately searchable/retrievable.
     */
    protected BulkResponse grpcIndexDocument(ManagedChannel channel, String index, String id, ByteString sourceBytes) {
        IndexOperation indexOp = IndexOperation.newBuilder().setXIndex(index).setXId(id).build();
        BulkRequestBody body = BulkRequestBody.newBuilder()
            .setOperationContainer(OperationContainer.newBuilder().setIndex(indexOp).build())
            .setObject(sourceBytes)
            .build();
        BulkRequest request = BulkRequest.newBuilder().setRefresh(Refresh.REFRESH_TRUE).addBulkRequestBody(body).build();
        DocumentServiceGrpc.DocumentServiceBlockingStub stub = DocumentServiceGrpc.newBlockingStub(channel)
            .withDeadlineAfter(30, TimeUnit.SECONDS);
        return stub.bulk(request);
    }
}
