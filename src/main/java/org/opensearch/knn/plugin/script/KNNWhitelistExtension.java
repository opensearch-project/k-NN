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
/*
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
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
