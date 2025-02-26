/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.apache.hc.client5.http.impl.DefaultHttpRequestRetryStrategy;
import org.apache.hc.core5.http.ConnectionClosedException;
import org.apache.hc.core5.util.TimeValue;

import javax.net.ssl.SSLException;
import java.io.InterruptedIOException;
import java.net.ConnectException;
import java.net.NoRouteToHostException;
import java.net.UnknownHostException;
import java.util.List;

/**
 * The public constructors for the Apache HTTP client default retry strategies allow customization of max retries
 * and retry interval, but not retryable status codes.
 * In order to add the other retryable status codes from our Remote Build API Contract, we must extend this class.
 * @see org.apache.hc.client5.http.impl.DefaultHttpRequestRetryStrategy
 */
public class RemoteIndexClientRetryStrategy extends DefaultHttpRequestRetryStrategy {
    private static final List<Integer> retryableCodes = List.of(408, 429, 500, 502, 503, 504, 509);

    public RemoteIndexClientRetryStrategy() {
        super(
            RemoteIndexHTTPClient.MAX_RETRIES,
            TimeValue.ofMilliseconds(RemoteIndexHTTPClient.BASE_DELAY_MS),
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
}
