/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.hc.client5.http.impl.nio.PoolingAsyncClientConnectionManager;
import org.apache.hc.client5.http.impl.nio.PoolingAsyncClientConnectionManagerBuilder;
import org.apache.hc.client5.http.ssl.ClientTlsStrategyBuilder;
import org.apache.hc.client5.http.ssl.NoopHostnameVerifier;
import org.apache.hc.core5.http.Header;
import org.apache.hc.core5.http.HttpHost;
import org.apache.hc.client5.http.auth.AuthScope;
import org.apache.hc.client5.http.auth.UsernamePasswordCredentials;
import org.apache.hc.client5.http.impl.auth.BasicCredentialsProvider;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.message.BasicHeader;
import org.apache.hc.core5.http.nio.ssl.TlsStrategy;
import org.apache.hc.core5.reactor.ssl.TlsDetails;
import org.apache.hc.core5.ssl.SSLContextBuilder;
import org.apache.hc.core5.util.Timeout;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.RestClient;
import org.opensearch.client.RestClientBuilder;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.search.SearchHit;
import org.opensearch.test.rest.OpenSearchRestTestCase;
import org.junit.After;

import static org.opensearch.knn.TestUtils.KNN_BWC_PREFIX;
import static org.opensearch.knn.TestUtils.OPENDISTRO_SECURITY;
import static org.opensearch.knn.TestUtils.ML_PLUGIN_SYSTEM_INDEX_PREFIX;
import static org.opensearch.knn.TestUtils.OPENSEARCH_SYSTEM_INDEX_PREFIX;
import static org.opensearch.knn.TestUtils.SECURITY_AUDITLOG_PREFIX;
import static org.opensearch.knn.TestUtils.SKIP_DELETE_MODEL_INDEX;
import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.TASKS_INDEX_NAME;

/**
 * ODFE integration test base class to support both security disabled and enabled ODFE cluster.
 */
public abstract class ODFERestTestCase extends OpenSearchRestTestCase {

    private final Set<String> IMMUTABLE_INDEX_PREFIXES = Set.of(
        KNN_BWC_PREFIX,
        SECURITY_AUDITLOG_PREFIX,
        OPENSEARCH_SYSTEM_INDEX_PREFIX,
        ML_PLUGIN_SYSTEM_INDEX_PREFIX
    );

    protected boolean isHttps() {
        return Optional.ofNullable(System.getProperty("https")).map("true"::equalsIgnoreCase).orElse(false);
    }

    @Override
    protected String getProtocol() {
        return isHttps() ? "https" : "http";
    }

    @Override
    protected RestClient buildClient(Settings settings, HttpHost[] hosts) throws IOException {
        RestClientBuilder builder = RestClient.builder(hosts);
        if (isHttps()) {
            configureHttpsClient(builder, settings);
        } else {
            configureClient(builder, settings);
        }

        builder.setStrictDeprecationMode(false);
        return builder.build();
    }

    protected static void configureHttpsClient(RestClientBuilder builder, Settings settings) throws IOException {
        // Similar to client configuration with OpenSearch:
        // https://github.com/opensearch-project/OpenSearch/blob/2.11.1/test/framework/src/main/java/org/opensearch/test/rest/OpenSearchRestTestCase.java#L841-L863
        // except we set the user name and password
        builder.setHttpClientConfigCallback(httpClientBuilder -> {
            String userName = Optional.ofNullable(System.getProperty("user"))
                .orElseThrow(() -> new RuntimeException("user name is missing"));
            String password = Optional.ofNullable(System.getProperty("password"))
                .orElseThrow(() -> new RuntimeException("password is missing"));
            BasicCredentialsProvider credentialsProvider = new BasicCredentialsProvider();
            final AuthScope anyScope = new AuthScope(null, -1);
            credentialsProvider.setCredentials(anyScope, new UsernamePasswordCredentials(userName, password.toCharArray()));
            try {
                final TlsStrategy tlsStrategy = ClientTlsStrategyBuilder.create()
                    .setHostnameVerifier(NoopHostnameVerifier.INSTANCE)
                    .setSslContext(SSLContextBuilder.create().loadTrustMaterial(null, (chains, authType) -> true).build())
                    // See https://issues.apache.org/jira/browse/HTTPCLIENT-2219
                    .setTlsDetailsFactory(sslEngine -> new TlsDetails(sslEngine.getSession(), sslEngine.getApplicationProtocol()))
                    .build();
                final PoolingAsyncClientConnectionManager connectionManager = PoolingAsyncClientConnectionManagerBuilder.create()
                    .setTlsStrategy(tlsStrategy)
                    .build();
                return httpClientBuilder.setDefaultCredentialsProvider(credentialsProvider).setConnectionManager(connectionManager);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        Map<String, String> headers = ThreadContext.buildDefaultHeaders(settings);
        Header[] defaultHeaders = new Header[headers.size()];
        int i = 0;
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            defaultHeaders[i++] = new BasicHeader(entry.getKey(), entry.getValue());
        }
        builder.setDefaultHeaders(defaultHeaders);
        final String socketTimeoutString = settings.get(CLIENT_SOCKET_TIMEOUT);
        final TimeValue socketTimeout = TimeValue.parseTimeValue(
            socketTimeoutString == null ? "60s" : socketTimeoutString,
            CLIENT_SOCKET_TIMEOUT
        );
        builder.setRequestConfigCallback(
            conf -> conf.setResponseTimeout(Timeout.ofMilliseconds(Math.toIntExact(socketTimeout.getMillis())))
        );
        if (settings.hasValue(CLIENT_PATH_PREFIX)) {
            builder.setPathPrefix(settings.get(CLIENT_PATH_PREFIX));
        }
    }

    /**
     * wipeAllIndices won't work since it cannot delete security index. Use wipeAllODFEIndices instead.
     */
    @Override
    protected boolean preserveIndicesUponCompletion() {
        return true;
    }

    @SuppressWarnings("unchecked")
    @After
    protected void wipeAllODFEIndices() throws Exception {
        Response response = adminClient().performRequest(new Request("GET", "/_cat/indices?format=json&expand_wildcards=all"));
        MediaType mediaType = MediaType.fromMediaType(response.getEntity().getContentType());
        try (
            XContentParser parser = mediaType.xContent()
                .createParser(
                    NamedXContentRegistry.EMPTY,
                    DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                    response.getEntity().getContent()
                )
        ) {
            XContentParser.Token token = parser.nextToken();
            List<Map<String, Object>> parserList = null;
            if (token == XContentParser.Token.START_ARRAY) {
                parserList = parser.listOrderedMap().stream().map(obj -> (Map<String, Object>) obj).collect(Collectors.toList());
            } else {
                parserList = Collections.singletonList(parser.mapOrdered());
            }

            for (Map<String, Object> index : parserList) {
                final String indexName = (String) index.get("index");
                if (MODEL_INDEX_NAME.equals(indexName)) {
                    if (!getSkipDeleteModelIndexFlag()) {
                        deleteModels(getModelIds());
                    }
                    continue;
                }
                if (!skipDeleteIndex(indexName)) {
                    adminClient().performRequest(new Request("DELETE", "/" + indexName));
                }
            }
        }
    }

    private List<String> getModelIds() throws IOException, ParseException {
        final String restURIGetModels = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, "_search");
        final Response response = adminClient().performRequest(new Request("GET", restURIGetModels));

        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        final String responseBody = EntityUtils.toString(response.getEntity());
        assertNotNull(responseBody);

        final XContentParser parser = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody);
        final SearchResponse searchResponse = SearchResponse.fromXContent(parser);

        return Arrays.stream(searchResponse.getHits().getHits()).map(SearchHit::getId).collect(Collectors.toList());
    }

    private void deleteModels(final List<String> modelIds) throws IOException {
        for (final String testModelID : modelIds) {
            final String restURIGetModel = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, testModelID);
            final Response getModelResponse = adminClient().performRequest(new Request("GET", restURIGetModel));
            if (RestStatus.OK != RestStatus.fromCode(getModelResponse.getStatusLine().getStatusCode())) {
                continue;
            }
            final String restURIDeleteModel = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, testModelID);
            adminClient().performRequest(new Request("DELETE", restURIDeleteModel));
        }
    }

    private boolean getSkipDeleteModelIndexFlag() {
        return Boolean.parseBoolean(System.getProperty(SKIP_DELETE_MODEL_INDEX, "false"));
    }

    private boolean skipDeleteIndex(String indexName) {
        return indexName == null
            || OPENDISTRO_SECURITY.equals(indexName)
            || IMMUTABLE_INDEX_PREFIXES.stream().anyMatch(indexName::startsWith)
            || MODEL_INDEX_NAME.equals(indexName)
            || TASKS_INDEX_NAME.equals(indexName);
    }
}
