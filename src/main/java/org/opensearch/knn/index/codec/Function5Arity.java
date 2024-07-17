/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

@FunctionalInterface
public interface Function5Arity<S, T, U, V, X, R> {
    R apply(S s, T t, U u, V v, X x);
}
