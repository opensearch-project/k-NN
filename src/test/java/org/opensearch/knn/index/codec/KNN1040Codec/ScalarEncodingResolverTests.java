/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.knn.KNNTestCase;

public class ScalarEncodingResolverTests extends KNNTestCase {

    public void testForDocBits_returnsExpectedEncoding() {
        assertEquals(ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE, ScalarEncodingResolver.forDocBits(1));
        assertEquals(ScalarEncoding.DIBIT_QUERY_NIBBLE, ScalarEncodingResolver.forDocBits(2));
        assertEquals(ScalarEncoding.PACKED_NIBBLE, ScalarEncodingResolver.forDocBits(4));
    }

    public void testForDocBits_unsupportedBits_throws() {
        for (int docBits : new int[] { 0, 3, 5, 7, 8, 16, -1 }) {
            IllegalArgumentException ex = expectThrows(IllegalArgumentException.class, () -> ScalarEncodingResolver.forDocBits(docBits));
            assertTrue(
                "Expected message to mention the unsupported bit width, got: " + ex.getMessage(),
                ex.getMessage().contains(String.valueOf(docBits))
            );
            assertTrue("Expected message to list supported widths, got: " + ex.getMessage(), ex.getMessage().contains("[1, 2, 4]"));
        }
    }

    public void testDocBits_isInverseOfForDocBits() {
        for (int docBits : new int[] { 1, 2, 4 }) {
            ScalarEncoding encoding = ScalarEncodingResolver.forDocBits(docBits);
            assertEquals(docBits, ScalarEncodingResolver.docBits(encoding));
        }
    }

    public void testDocBits_rejectsUnsupportedEncodings() {
        // Encodings whose bit widths are not in SUPPORTED_DOC_BITS must fail loud so that unexpected
        // Lucene encodings can never leak into downstream code that assumes bits ∈ {1, 2, 4}.
        for (ScalarEncoding encoding : new ScalarEncoding[] { ScalarEncoding.SEVEN_BIT, ScalarEncoding.UNSIGNED_BYTE }) {
            IllegalArgumentException ex = expectThrows(IllegalArgumentException.class, () -> ScalarEncodingResolver.docBits(encoding));
            assertTrue("Expected message to name the encoding, got: " + ex.getMessage(), ex.getMessage().contains(encoding.name()));
            assertTrue("Expected message to list supported widths, got: " + ex.getMessage(), ex.getMessage().contains("[1, 2, 4]"));
        }
    }

    public void testDocBits_nullEncoding_throwsNPE() {
        expectThrows(NullPointerException.class, () -> ScalarEncodingResolver.docBits(null));
    }

    public void testQueryBits_isFourForAllSupportedEncodings() {
        assertEquals(4, ScalarEncodingResolver.QUERY_BITS);
        for (int docBits : new int[] { 1, 2, 4 }) {
            ScalarEncoding encoding = ScalarEncodingResolver.forDocBits(docBits);
            assertEquals(
                "Encoding " + encoding + " should use " + ScalarEncodingResolver.QUERY_BITS + "-bit query",
                ScalarEncodingResolver.QUERY_BITS,
                encoding.getQueryBits()
            );
        }
    }
}
