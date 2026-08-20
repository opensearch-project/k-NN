/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.test.OpenSearchTestCase;

import java.util.HashMap;
import java.util.Map;

public class EngineParametersTests extends OpenSearchTestCase {

    private static final ParameterKey<Integer> WINDOW = ParameterKey.intKey("window");
    private static final ParameterKey<String> MODE = ParameterKey.stringKey("mode");

    public void testTypedReadAndDefault() {
        final EngineParameters params = EngineParameters.of(Map.of("window", 7));
        assertEquals(Integer.valueOf(7), params.get(WINDOW));
        assertEquals(Integer.valueOf(7), params.get(WINDOW, 99));
        assertNull(params.get(MODE));
        assertEquals("exact", params.get(MODE, "exact"));
        assertTrue(params.has(WINDOW));
        assertFalse(params.has(MODE));
    }

    public void testNumberWidening() {
        // Parsers hand over the narrowest Number they chose, the key widens it.
        final EngineParameters params = EngineParameters.of(Map.of("window", (short) 7));
        assertEquals(Integer.valueOf(7), params.get(WINDOW));
    }

    public void testTypeMismatchThrows() {
        final EngineParameters params = EngineParameters.of(Map.of("window", "not a number"));
        final IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> params.get(WINDOW));
        assertTrue(e.getMessage().contains("window"));
        assertTrue(e.getMessage().contains("Integer"));
    }

    public void testNullAndEmptyMapsShareTheEmptyInstance() {
        assertSame(EngineParameters.EMPTY, EngineParameters.of(null));
        assertSame(EngineParameters.EMPTY, EngineParameters.of(Map.of()));
    }

    public void testRawIsUnmodifiable() {
        final Map<String, Object> source = new HashMap<>(Map.of("window", 7));
        final EngineParameters params = EngineParameters.of(source);
        expectThrows(UnsupportedOperationException.class, () -> params.raw().put("k", 1));
        // And a later change to the source map is not visible, the view copied.
        source.put("window", 8);
        assertEquals(Integer.valueOf(7), params.get(WINDOW));
    }
}
