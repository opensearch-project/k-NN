/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;

import java.io.IOException;

import static org.mockito.Mockito.mock;

public class InternalNestedKnnVectorQueryTests extends TestCase {

    /**
     * The default {@link InternalNestedKnnVectorQuery#knnRescoreSearch} must throw
     * {@link UnsupportedOperationException} for implementations that never enter the rescore path
     * (only the float implementation supports it). This guards against a silent no-op if a new
     * implementation is ever routed through the rescore step without providing its own override.
     */
    public void testKnnRescoreSearch_whenNotOverridden_thenThrowsUnsupportedOperation() {
        // Implementation that leaves knnRescoreSearch as the interface default.
        InternalNestedKnnVectorQuery query = new InternalNestedKnnVectorQuery() {
            @Override
            public Query knnRewrite(final IndexSearcher searcher) {
                return null;
            }

            @Override
            public TopDocs knnExactSearch(final LeafReaderContext leafReaderContext, final DocIdSetIterator iterator) {
                return null;
            }

            @Override
            public String getField() {
                return "field";
            }

            @Override
            public Query getFilter() {
                return null;
            }

            @Override
            public int getK() {
                return 0;
            }

            @Override
            public BitSetProducer getParentFilter() {
                return null;
            }
        };

        LeafReaderContext context = mock(LeafReaderContext.class);
        DocIdSetIterator iterator = mock(DocIdSetIterator.class);
        try {
            query.knnRescoreSearch(context, iterator);
            fail("Expected UnsupportedOperationException from the default knnRescoreSearch implementation");
        } catch (UnsupportedOperationException e) {
            assertEquals("Rescore search is not supported for this query type", e.getMessage());
        } catch (IOException e) {
            fail("Unexpected IOException: " + e.getMessage());
        }
    }
}
