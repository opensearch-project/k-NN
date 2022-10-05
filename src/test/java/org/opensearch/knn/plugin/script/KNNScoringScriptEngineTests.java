/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.io.IOException;
import java.util.Set;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;

public class KNNScoringScriptEngineTests extends KNNTestCase {

    public void testGetSupportedContexts() throws IOException {
        try (ScriptEngine engine = new KNNScoringScriptEngine()) {
            Set<ScriptContext<?>> supportedContexts = engine.getSupportedContexts();
            assertNotNull(supportedContexts);
            assertFalse(supportedContexts.isEmpty());
        }
    }

}
