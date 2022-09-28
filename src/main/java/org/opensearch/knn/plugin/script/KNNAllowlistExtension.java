/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import com.google.common.collect.ImmutableMap;
import org.opensearch.painless.spi.PainlessExtension;
import org.opensearch.painless.spi.Allowlist;
import org.opensearch.painless.spi.AllowlistLoader;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptedMetricAggContexts;

import java.util.List;
import java.util.Map;

public class KNNAllowlistExtension implements PainlessExtension {

    private static final Allowlist ALLOW_LIST = AllowlistLoader.loadFromResourceFiles(KNNAllowlistExtension.class, "knn_allowlist.txt");

    @Override
    public Map<ScriptContext<?>, List<Allowlist>> getContextAllowlists() {
        final List<Allowlist> allowLists = List.of(ALLOW_LIST);
        return ImmutableMap.of(
            ScoreScript.CONTEXT,
            allowLists,
            ScriptedMetricAggContexts.InitScript.CONTEXT,
            allowLists,
            ScriptedMetricAggContexts.MapScript.CONTEXT,
            allowLists,
            ScriptedMetricAggContexts.CombineScript.CONTEXT,
            allowLists,
            ScriptedMetricAggContexts.ReduceScript.CONTEXT,
            allowLists
        );
    }
}
