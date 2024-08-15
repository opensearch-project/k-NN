/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.AllArgsConstructor;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Arrays;
import java.util.Collection;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;

@AllArgsConstructor
public class RescoreValidationTests extends KNNTestCase {

    private boolean isValid;
    private RescoreContext rescoreContext;

    @ParametersFactory(argumentFormatting = "isValid:%1$s; rescoreContext:%2$s")
    public static Collection<Object[]> validParams() {
        return Arrays.asList(
            $$(
                $(true, RescoreContext.builder().build()),
                $(true, RescoreContext.getDefault()),
                $(true, RescoreContext.builder().oversampleFactor(RescoreContext.MAX_OVERSAMPLE_FACTOR - 1).build()),
                $(false, RescoreContext.builder().oversampleFactor(RescoreContext.MAX_OVERSAMPLE_FACTOR + 1).build()),
                $(false, RescoreContext.builder().oversampleFactor(RescoreContext.MIN_OVERSAMPLE_FACTOR - 1).build())
            )
        );
    }

    public void testValidate() {
        if (isValid) {
            assertNull(RescoreParser.validate(rescoreContext));
        } else {
            assertNotNull(RescoreParser.validate(rescoreContext));
        }
    }
}
