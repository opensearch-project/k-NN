/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;

import java.lang.reflect.Method;

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

    @SneakyThrows
    public static <T> T triggerDoLoad(final IndexInput input, T index) {
        final Method doLoadMethod = FaissIdMapIndex.class.getDeclaredMethod("doLoad", IndexInput.class);
        doLoadMethod.setAccessible(true);
        doLoadMethod.invoke(index, input);
        return index;
    }
}
