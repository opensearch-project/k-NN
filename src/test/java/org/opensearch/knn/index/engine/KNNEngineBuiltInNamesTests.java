/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.test.OpenSearchTestCase;

import java.util.Locale;

public class KNNEngineBuiltInNamesTests extends OpenSearchTestCase {

    public void testEveryBuiltInNameIsReserved() {
        // A built-in missing from the reserved set could be silently shadowed by a registered engine of
        // the same name. In the default build values() is exactly the built-ins.
        for (KNNEngine engine : KNNEngine.values()) {
            assertTrue(
                "Built-in engine [" + engine.getName() + "] must be in KNNEngineRegistry.BUILT_IN_ENGINE_NAMES",
                KNNEngineRegistry.BUILT_IN_ENGINE_NAMES.contains(engine.getName().toLowerCase(Locale.ROOT))
            );
        }
    }
}
