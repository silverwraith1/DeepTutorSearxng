# -*- coding: utf-8 -*-
"""
SearXNG Metasearch Provider
"""

import os
from datetime import datetime
from typing import Any, List, Dict

from langchain_community.utilities import SearxSearchWrapper
from ..base import BaseSearchProvider
from ..types import Citation, SearchResult, WebSearchResponse
from . import register_provider

@register_provider("searxng")
class SearxNGProvider(BaseSearchProvider):
    """SearXNG metasearch provider using LangChain wrapper"""

    display_name = "SearXNG"
    description = "Self-hosted metasearch engine"
    supports_answer = False  # SearXNG returns raw results

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Pulls from your environment variable SEARXNG_URL
        self.searx_host = os.environ.get("SEARXNG_URL", "http://localhost:8080")
        self.client = SearxSearchWrapper(searx_host=self.searx_host)

    def search(
        self,
        query: str,
        num_results: int = 10,
        engines: List[str] = None,
        categories: List[str] = None,
        **kwargs: Any,
    ) -> WebSearchResponse:
        """
        Perform search using local SearXNG instance.
        """
        self.logger.debug(f"Calling SearXNG at {self.searx_host} for query: {query}")
        
        # Get structured results from LangChain
        raw_results = self.client.results(
            query, 
            num_results=num_results, 
            engines=engines, 
            categories=categories
        )

        citations: List[Citation] = []
        search_results: List[SearchResult] = []

        for i, result in enumerate(raw_results, 1):
            title = result.get("title", "No Title")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            
            # 1. Create the SearchResult object
            sr = SearchResult(
                title=title,
                url=link,
                snippet=snippet,
                source="searxng",
                date="",  # SearXNG doesn't always provide a specific date string
                sitelinks=[],
                attributes={}
            )
            search_results.append(sr)

            # 2. Create the Citation object (required for the UI to show [1], [2], etc.)
            citations.append(
                Citation(
                    id=i,
                    reference=f"[{i}]",
                    url=link,
                    title=title,
                    snippet=snippet,
                    source="searxng"
                )
            )

        # 3. Return the standardized response the ChatAgent expects
        return WebSearchResponse(
            query=query,
            answer="", # SearXNG is a searcher, not a generator
            provider="searxng",
            timestamp=datetime.now().isoformat(),
            model="searxng-local",
            citations=citations,
            search_results=search_results,
            usage={},
            metadata={"searx_host": self.searx_host}
        )
