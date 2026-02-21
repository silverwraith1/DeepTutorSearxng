# -*- coding: utf-8 -*-
"""
SearxNG Search Provider
"""

import logging
from datetime import datetime
from typing import Any

from ..base import BaseSearchProvider
from ..types import Citation, SearchResult, WebSearchResponse
from . import register_provider

logger = logging.getLogger(__name__)


@register_provider("searxng")
class SearxNGProvider(BaseSearchProvider):
    """SearxNG search provider implementation"""

    display_name = "SearxNG"
    description = "Privacy-respecting metasearch engine"
    supports_answer = False

    def search(
        self,
        query: str,
        num: int = 10,
        **kwargs: Any,
    ) -> WebSearchResponse:
        """
        Perform search using SearxNG via LangChain wrapper.
        """
        # Lazy import to prevent environment errors during linting
        try:
            from langchain_community.utilities import SearxSearchWrapper
        except ImportError:
            logger.error("langchain-community not installed")
            raise ImportError("Please install langchain-community to use SearxNG")

        # Initialize wrapper using config from self (BaseSearchProvider handles config)
        # Note: self.api_key or self.config is available depending on base class setup
        host = kwargs.get("host") or getattr(self, "host", None)
        
        wrapper = SearxSearchWrapper(
            searx_host=host,
            engines=kwargs.get("engines", []),
        )

        raw_results = wrapper.results(query, num_results=num)

        citations: list[Citation] = []
        search_results: list[SearchResult] = []

        for i, result in enumerate(raw_results, 1):
            title = result.get("title", "")
            url_val = result.get("link", "")
            snippet = result.get("snippet", "")

            sr = SearchResult(
                title=title,
                url=url_val,
                snippet=snippet,
                source="searxng",
            )
            search_results.append(sr)

            citations.append(
                Citation(
                    id=i,
                    reference=f"[{i}]",
                    url=url_val,
                    title=title,
                    snippet=snippet,
                    source="searxng",
                )
            )

        return WebSearchResponse(
            query=query,
            answer="",
            provider="searxng",
            timestamp=datetime.now().isoformat(),
            model="searxng",
            citations=citations,
            search_results=search_results,
            usage={},
            metadata={},
        )
