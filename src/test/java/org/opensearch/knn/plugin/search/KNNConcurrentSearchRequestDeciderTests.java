/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.search;

import org.opensearch.index.IndexSettings;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.search.deciders.ConcurrentSearchDecision;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNConcurrentSearchRequestDeciderTests extends KNNTestCase {

    public void testDecider_thenSucceed() {
        ConcurrentSearchDecision noop = new ConcurrentSearchDecision(ConcurrentSearchDecision.DecisionStatus.NO_OP, "Default decision");

        KNNConcurrentSearchRequestDecider decider = new KNNConcurrentSearchRequestDecider();
        assertDecision(noop, decider.getConcurrentSearchDecision());
        IndexSettings indexSettingsMock = mock(IndexSettings.class);
        when(indexSettingsMock.getValue(KNNSettings.IS_KNN_INDEX_SETTING)).thenReturn(Boolean.FALSE);

        // Non KNNQueryBuilder
        decider.evaluateForQuery(new MatchAllQueryBuilder(), indexSettingsMock);
        assertDecision(noop, decider.getConcurrentSearchDecision());
        decider.evaluateForQuery(
            KNNQueryBuilder.builder().vector(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }).fieldName("decider").k(10).build(),
            indexSettingsMock
        );
        assertDecision(noop, decider.getConcurrentSearchDecision());

        when(indexSettingsMock.getValue(KNNSettings.IS_KNN_INDEX_SETTING)).thenReturn(Boolean.TRUE);
        decider.evaluateForQuery(
            KNNQueryBuilder.builder().vector(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }).fieldName("decider").k(10).build(),
            indexSettingsMock
        );
        ConcurrentSearchDecision yes = new ConcurrentSearchDecision(
            ConcurrentSearchDecision.DecisionStatus.YES,
            "Enable concurrent search for knn as Query has k-NN query in it and index is k-nn index"
        );
        assertDecision(yes, decider.getConcurrentSearchDecision());

        decider.evaluateForQuery(new MatchAllQueryBuilder(), indexSettingsMock);
        assertDecision(noop, decider.getConcurrentSearchDecision());
    }

    public void testDeciderFactory_thenSucceed() {
        KNNConcurrentSearchRequestDecider.Factory factory = new KNNConcurrentSearchRequestDecider.Factory();
        IndexSettings indexSettingsMock = mock(IndexSettings.class);
        when(indexSettingsMock.getValue(KNNSettings.IS_KNN_INDEX_SETTING)).thenReturn(Boolean.TRUE);
        assertNotSame(factory.create(indexSettingsMock).get(), factory.create(indexSettingsMock).get());
        when(indexSettingsMock.getValue(KNNSettings.IS_KNN_INDEX_SETTING)).thenReturn(Boolean.FALSE);
        assertTrue(factory.create(indexSettingsMock).isEmpty());
    }

    private void assertDecision(ConcurrentSearchDecision expected, ConcurrentSearchDecision actual) {
        assertEquals(expected.getDecisionReason(), actual.getDecisionReason());
        assertEquals(expected.getDecisionStatus(), actual.getDecisionStatus());
    }
}
