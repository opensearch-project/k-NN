/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.SegmentReadState;

import java.io.IOException;

@FunctionalInterface
public interface DerivedSourceReaderSupplier<R> {
    R apply(SegmentReadState segmentReadState) throws IOException;
}
