/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.io.IOException;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.script.ScriptEngine;

public class KNNScoringScriptEngineTests extends KNNTestCase {

    public void testGetSupportedContexts() throws IOException {
        try (ScriptEngine engine = new KNNScoringScriptEngine()) {
            assertNotNull(engine.getSupportedContexts());
        }
    }

}
