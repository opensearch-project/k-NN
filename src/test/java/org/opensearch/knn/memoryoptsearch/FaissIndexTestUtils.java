/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.store.ByteBuffersDataOutput;

public class FaissIndexTestUtils {
    public static byte[] makeCommonHeader(final int dimension, final long totalNumberOfVectors, final boolean l2MetricType) {
        final ByteBuffersDataOutput output = new ByteBuffersDataOutput();
        // Dimension
        output.writeInt(dimension);
        // #vectorrs
        output.writeLong(totalNumberOfVectors);

        // Write two dummy fields
        output.writeLong(0);
        output.writeLong(0);

        // isTrained = False
        output.writeByte((byte) 0);

        // Metric type
        output.writeInt(l2MetricType ? 1 : 0);

        return output.toArrayCopy();
    }

    public static byte[] makeBinaryCommonHeader(final int dimension, final int codeSize, final long totalNumberOfVectors) {
        final ByteBuffersDataOutput output = new ByteBuffersDataOutput();
        // Dimension
        output.writeInt(dimension);
        // Code size
        output.writeInt(codeSize);
        // #vectors
        output.writeLong(totalNumberOfVectors);

        // Two dummy fields: is_trained, metric_type
        output.writeByte((byte) 0);
        output.writeInt((byte) 0);

        return output.toArrayCopy();
    }
}
