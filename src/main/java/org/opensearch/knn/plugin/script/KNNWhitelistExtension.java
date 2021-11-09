/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.painless.spi.PainlessExtension;
import org.opensearch.painless.spi.Whitelist;
import org.opensearch.painless.spi.WhitelistLoader;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class KNNWhitelistExtension implements PainlessExtension {

    private static final Whitelist WHITELIST =
            WhitelistLoader.loadFromResourceFiles(KNNWhitelistExtension.class, "knn_whitelist.txt");

    @Override
    public Map<ScriptContext<?>, List<Whitelist>> getContextWhitelists() {
        return Collections.singletonMap(ScoreScript.CONTEXT, Collections.singletonList(WHITELIST));
    }
}
