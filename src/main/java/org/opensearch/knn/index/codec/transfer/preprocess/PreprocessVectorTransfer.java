/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer.preprocess;

import org.opensearch.knn.index.codec.transfer.VectorTransfer;

import java.util.function.Function;

@FunctionalInterface
public interface PreprocessVectorTransfer<T, V> extends Function<T, V> {
}
