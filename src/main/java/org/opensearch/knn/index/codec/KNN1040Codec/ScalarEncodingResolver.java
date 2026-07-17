/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Single decision point that maps a document bit width (the number of bits used to quantize each
 * dimension of a stored vector) to the Lucene {@link ScalarEncoding} used by the FAISS SQ
 * memory-optimized-search path, and back.
 *
 * <p>The FAISS SQ path always uses a 4-bit (nibble) query, so the relevant encodings are exactly
 * those whose query bit width is {@link #QUERY_BITS}. In Lucene 10.4 these are:
 * <ul>
 *   <li>1-bit document → {@code SINGLE_BIT_QUERY_NIBBLE} (x32)</li>
 *   <li>2-bit document → {@code DIBIT_QUERY_NIBBLE} (x16)</li>
 *   <li>4-bit document → {@code PACKED_NIBBLE} (x8)</li>
 * </ul>
 *
 * <p>The mapping is resolved dynamically from {@link ScalarEncoding#getBits()} and
 * {@link ScalarEncoding#getQueryBits()} rather than by hardcoding enum names, so it stays correct
 * if Lucene renames constants and so adding a new supported bit width only requires updating
 * {@link #SUPPORTED_DOC_BITS}. This keeps the {@code SINGLE_BIT_QUERY_NIBBLE} reference out of the
 * codec format, the build strategy, and the scorer (Requirement 11.4).
 */
public final class ScalarEncodingResolver {

    private ScalarEncodingResolver() {
        // Utility class; not instantiable.
    }

    /** The FAISS SQ path always quantizes the query to a 4-bit nibble. */
    public static final int QUERY_BITS = 4;

    /** Document bit widths supported by the FAISS SQ memory-optimized-search path. */
    private static final int[] SUPPORTED_DOC_BITS = { 1, 2, 4 };

    /** docBits -> ScalarEncoding, resolved once from the enum metadata. */
    private static final Map<Integer, ScalarEncoding> DOC_BITS_TO_ENCODING = buildDocBitsToEncoding();

    /**
     * Builds the docBits → {@link ScalarEncoding} lookup table by matching entries in
     * {@link ScalarEncoding#values()} against each supported width paired with the fixed
     * {@link #QUERY_BITS} (4). Called once at class-load; the returned map is stored in
     * {@link #DOC_BITS_TO_ENCODING} and consulted for every {@link #forDocBits(int)} call.
     *
     * <p>Current mapping (Lucene 10.4, kept here for reference and to track deviations
     * over time — if this diverges, either Lucene renamed constants or the enum's
     * {@code getBits()}/{@code getQueryBits()} metadata changed):
     * <table>
     *   <caption>Current docBits → ScalarEncoding mapping</caption>
     *   <tr><th>docBits</th><th>ScalarEncoding</th><th>Compression</th></tr>
     *   <tr><td>1</td><td>{@code SINGLE_BIT_QUERY_NIBBLE}</td><td>x32</td></tr>
     *   <tr><td>2</td><td>{@code DIBIT_QUERY_NIBBLE}</td><td>x16</td></tr>
     *   <tr><td>4</td><td>{@code PACKED_NIBBLE}</td><td>x8</td></tr>
     * </table>
     *
     * <p>Throws at class-load time (via the static initializer of {@link #DOC_BITS_TO_ENCODING})
     * if any supported width is missing from the installed Lucene version, so upgrades that
     * silently drop an encoding fail fast rather than corrupting on-disk segments.
     */
    private static Map<Integer, ScalarEncoding> buildDocBitsToEncoding() {
        final Map<Integer, ScalarEncoding> map = new HashMap<>();
        for (int docBits : SUPPORTED_DOC_BITS) {
            for (ScalarEncoding encoding : ScalarEncoding.values()) {
                if (encoding.getBits() == docBits && encoding.getQueryBits() == QUERY_BITS) {
                    map.put(docBits, encoding);
                    break;
                }
            }
            if (map.containsKey(docBits) == false) {
                throw new IllegalStateException(
                    String.format(
                        Locale.ROOT,
                        "No Lucene ScalarEncoding found for document bits=%d with a %d-bit query. "
                            + "The installed Lucene version may not expose this encoding.",
                        docBits,
                        QUERY_BITS
                    )
                );
            }
        }
        return map;
    }

    /**
     * Returns the {@link ScalarEncoding} used to store documents quantized to {@code docBits} bits
     * per dimension with a 4-bit nibble query.
     *
     * @param docBits the document bit width (1, 2, or 4)
     * @return the corresponding Lucene scalar encoding
     * @throws IllegalArgumentException if {@code docBits} is not a supported document bit width
     */
    public static ScalarEncoding forDocBits(final int docBits) {
        final ScalarEncoding encoding = DOC_BITS_TO_ENCODING.get(docBits);
        if (encoding == null) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Unsupported SQ document bit width: %d. Supported: %s", docBits, supportedDocBitsString())
            );
        }
        return encoding;
    }

    /**
     * Inverse of {@link #forDocBits(int)} — returns the document bit width for {@code encoding},
     * rejecting encodings outside the {@link #SUPPORTED_DOC_BITS} set (1, 2, 4). {@link ScalarEncoding}
     * exposes bit widths for encodings the FAISS SQ path doesn't support (e.g. {@code SEVEN_BIT},
     * {@code UNSIGNED_BYTE}); validating here keeps the round-trip guarantee symmetric with
     * {@link #forDocBits(int)} and fails loud if a caller ever passes an unexpected encoding.
     *
     * @param encoding a Lucene scalar encoding
     * @return the document bit width ({@link ScalarEncoding#getBits()})
     * @throws NullPointerException if {@code encoding} is {@code null}
     * @throws IllegalArgumentException if {@code encoding} is not one of the supported FAISS SQ encodings
     */
    public static int docBits(final ScalarEncoding encoding) {
        Objects.requireNonNull(encoding, "ScalarEncoding must not be null");
        final int bits = encoding.getBits();
        if (DOC_BITS_TO_ENCODING.containsKey(bits) == false) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Unsupported SQ scalar encoding: %s (bits=%d). Supported bits: %s",
                    encoding,
                    bits,
                    supportedDocBitsString()
                )
            );
        }
        return bits;
    }

    private static String supportedDocBitsString() {
        return Arrays.toString(SUPPORTED_DOC_BITS);
    }
}
