/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.opensearch.knn.KNNTestCase;

public class DerivedVectorInjectionConsumerTests extends KNNTestCase {
    //
    // @SneakyThrows
    // public void testVectorInjection() {
    // FloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
    // List.of(new float[] { 1.0f, 2.0f }, new float[] { 2.0f, 3.0f }, new float[] { 3.0f, 4.0f }, new float[] { 4.0f, 5.0f })
    // );
    // final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);
    //
    // final XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
    // builder.field("test_text", "text-field");
    // builder.endObject();
    //
    // BytesReference bytesReference = BytesReference.bytes(builder);
    // toMap(bytesReference);
    //
    // DerivedVectorInjectionConsumer consumer = new DerivedVectorInjectionConsumer(Map.of("test_vector", () -> knnVectorValues));
    // logger.info(bytesReference.length());
    // byte[] modifiedBytes = consumer.apply(0, bytesReference.toBytesRef().bytes);
    // BytesReference modifiedBytesReference = BytesReference.fromByteBuffer(ByteBuffer.wrap(modifiedBytes));
    // toMap(modifiedBytesReference);
    //
    // modifiedBytes = consumer.apply(1, bytesReference.toBytesRef().bytes);
    // modifiedBytesReference = BytesReference.fromByteBuffer(ByteBuffer.wrap(modifiedBytes));
    // toMap(modifiedBytesReference);
    //
    // modifiedBytes = consumer.apply(0, bytesReference.toBytesRef().bytes);
    // modifiedBytesReference = BytesReference.fromByteBuffer(ByteBuffer.wrap(modifiedBytes));
    // toMap(modifiedBytesReference);
    //
    // fail("On purpose");
    // }
    //
    // private void toMap(BytesReference source) {
    // Tuple<? extends MediaType, Map<String, Object>> mapTuple = XContentHelper.convertToMap(source, true, MediaTypeRegistry.JSON);
    // logger.info(mapTuple.v2().toString());
    // }

}
