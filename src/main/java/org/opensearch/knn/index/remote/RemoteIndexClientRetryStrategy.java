/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.apache.hc.client5.http.impl.DefaultHttpRequestRetryStrategy;
import org.apache.hc.core5.http.ConnectionClosedException;
import org.apache.hc.core5.http.HttpResponse;
import org.apache.hc.core5.http.protocol.HttpContext;
import org.apache.hc.core5.util.TimeValue;

import javax.net.ssl.SSLException;
import java.io.InterruptedIOException;
import java.net.ConnectException;
import java.net.NoRouteToHostException;
import java.net.UnknownHostException;
import java.util.Arrays;

/**
 * The public constructors for the Apache HTTP client default retry strategies allow customization of max retries
 * and retry interval, but not retriable status codes.
 * In order to add the other retriable status codes from our Remote Build API Contract, we must extend this class.
 * @see org.apache.hc.client5.http.impl.DefaultHttpRequestRetryStrategy
 */
public class RemoteIndexClientRetryStrategy extends DefaultHttpRequestRetryStrategy {
    public RemoteIndexClientRetryStrategy() {
        super(
            RemoteIndexClient.MAX_RETRIES,
            TimeValue.ofMilliseconds(RemoteIndexClient.BASE_DELAY_MS),
            Arrays.asList(
                InterruptedIOException.class,
                UnknownHostException.class,
                ConnectException.class,
                ConnectionClosedException.class,
                NoRouteToHostException.class,
                SSLException.class
            ),
            Arrays.asList(408, 429, 500, 502, 503, 504, 509)
        );
    }

    /**
    * Override retry interval setting to implement backoff strategy. This is only relevant for future implementations where we may increase the retry count from 1 max retry.
    */
    @Override
    public TimeValue getRetryInterval(HttpResponse response, int execCount, HttpContext context) {
        if (response.getCode() == 429 || response.getCode() == 503) {
            long delay = RemoteIndexClient.BASE_DELAY_MS;
            long backoffDelay = delay * (long) Math.pow(2, execCount - 1);
            return TimeValue.ofMilliseconds(Math.min(backoffDelay, TimeValue.ofMinutes(1).toMilliseconds()));
        }
        return super.getRetryInterval(response, execCount, context);
    }
}
