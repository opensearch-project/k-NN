/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.client;

import org.apache.hc.client5.http.impl.DefaultHttpRequestRetryStrategy;
import org.apache.hc.client5.http.protocol.HttpClientContext;
import org.apache.hc.core5.http.ConnectionClosedException;
import org.apache.hc.core5.http.HttpRequest;
import org.apache.hc.core5.http.HttpResponse;
import org.apache.hc.core5.http.protocol.HttpContext;
import org.apache.hc.core5.util.TimeValue;

import javax.net.ssl.SSLException;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.ConnectException;
import java.net.NoRouteToHostException;
import java.net.UnknownHostException;
import java.util.List;

import static org.apache.hc.core5.http.HttpStatus.SC_BAD_GATEWAY;
import static org.apache.hc.core5.http.HttpStatus.SC_CONFLICT;
import static org.apache.hc.core5.http.HttpStatus.SC_INTERNAL_SERVER_ERROR;
import static org.apache.hc.core5.http.HttpStatus.SC_REQUEST_TIMEOUT;
import static org.apache.hc.core5.http.HttpStatus.SC_SERVICE_UNAVAILABLE;
import static org.apache.hc.core5.http.HttpStatus.SC_TOO_MANY_REQUESTS;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.BUILD_ENDPOINT;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.STATUS_ENDPOINT;
import static org.opensearch.remoteindexbuild.stats.RemoteIndexClientValue.BUILD_REQUEST_FAILURE_COUNT;
import static org.opensearch.remoteindexbuild.stats.RemoteIndexClientValue.STATUS_REQUEST_FAILURE_COUNT;

/**
 * The public constructors for the Apache HTTP client default retry strategies allow customization of max retries
 * and retry interval, but not retryable status codes.
 * In order to add the other retryable status codes from our Remote Build API Contract, we must extend this class.
 * TODO Future work: tune this retry strategy (MAX_RETRIES, BASE_DELAY_MS, exponential backoff/jitter) based on benchmarking
 * @see org.apache.hc.client5.http.impl.DefaultHttpRequestRetryStrategy
 */
public class RemoteIndexHTTPClientRetryStrategy extends DefaultHttpRequestRetryStrategy {
    private static final int SC_BANDWIDTH_LIMIT_EXCEEDED = 509;
    private static final int MAX_RETRIES = 1; // 2 total attempts
    private static final long BASE_DELAY_MS = 1000;

    private static final List<Integer> retryableCodes = List.of(
        SC_REQUEST_TIMEOUT,
        SC_TOO_MANY_REQUESTS,
        SC_INTERNAL_SERVER_ERROR,
        SC_BAD_GATEWAY,
        SC_SERVICE_UNAVAILABLE,
        SC_CONFLICT,
        SC_BANDWIDTH_LIMIT_EXCEEDED
    );

    public RemoteIndexHTTPClientRetryStrategy() {
        super(
            MAX_RETRIES,
            TimeValue.ofMilliseconds(BASE_DELAY_MS),
            List.of(
                InterruptedIOException.class,
                UnknownHostException.class,
                ConnectException.class,
                ConnectionClosedException.class,
                NoRouteToHostException.class,
                SSLException.class
            ),
            retryableCodes
        );
    }

    @Override
    public boolean retryRequest(HttpRequest request, IOException exception, int execCount, HttpContext context) {
        if (request.getRequestUri().endsWith(BUILD_ENDPOINT)) {
            BUILD_REQUEST_FAILURE_COUNT.increment();
        } else if (request.getRequestUri().endsWith(STATUS_ENDPOINT)) {
            STATUS_REQUEST_FAILURE_COUNT.increment();
        }
        return super.retryRequest(request, exception, execCount, context);
    }

    @Override
    public boolean retryRequest(HttpResponse response, int execCount, HttpContext context) {
        HttpRequest request = ((HttpClientContext) context).getRequest();
        if (request.getRequestUri().endsWith(BUILD_ENDPOINT)) {
            BUILD_REQUEST_FAILURE_COUNT.increment();
        } else if (request.getRequestUri().endsWith(STATUS_ENDPOINT)) {
            STATUS_REQUEST_FAILURE_COUNT.increment();
        }
        return super.retryRequest(response, execCount, context);
    }
}
