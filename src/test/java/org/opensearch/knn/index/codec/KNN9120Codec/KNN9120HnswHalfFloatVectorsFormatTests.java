/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN990Codec.halffloatcodec.KNN990HalfFloatFlatVectorsFormat;
import org.junit.Assert;
import lombok.SneakyThrows;
import java.lang.reflect.Field;

public class KNN9120HnswHalfFloatVectorsFormatTests extends KNNTestCase {

    @SneakyThrows
    public void testDefaultConstructor() {
        KNN9120HnswHalfFloatVectorsFormat format = new KNN9120HnswHalfFloatVectorsFormat();
        Assert.assertNotNull(format);
    }

    @SneakyThrows
    public void testConstructorWithParams() {
        KNN9120HnswHalfFloatVectorsFormat format = new KNN9120HnswHalfFloatVectorsFormat(16, 32);
        Assert.assertNotNull(format);
    }

    @SneakyThrows
    public void testToString() {
        KNN9120HnswHalfFloatVectorsFormat format = new KNN9120HnswHalfFloatVectorsFormat(16, 32);
        String str = format.toString();
        Assert.assertTrue(str.contains("KNN9120HnswHalfFloatVectorsFormat"));
        Assert.assertTrue(str.contains("maxConn=16"));
        Assert.assertTrue(str.contains("beamWidth=32"));
    }

    @SneakyThrows
    public void testGetMaxDimensions() {
        KNN9120HnswHalfFloatVectorsFormat format = new KNN9120HnswHalfFloatVectorsFormat();
        int maxDim = format.getMaxDimensions("testField");
        Assert.assertTrue(maxDim > 0);
    }

    @SneakyThrows
    public void testFlatVectorsFormatIsHalfFloat() throws Exception {
        Field field = KNN9120HnswHalfFloatVectorsFormat.class.getDeclaredField("flatVectorsFormat");
        field.setAccessible(true);
        Object formatInstance = field.get(null);
        Assert.assertTrue(formatInstance instanceof KNN990HalfFloatFlatVectorsFormat);
    }
}
