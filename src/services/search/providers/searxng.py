import logging
from typing import Any, Dict, List, Optional

from pydantic import Field
from src.services.search.providers.base import BaseSearchProvider

logger = logging.getLogger(__name__)


class SearxNGProvider(BaseSearchProvider):
    """SearxNG search provider with full error handling and configuration."""

    name: str = "searxng"
    host: str = Field(..., description="The SearxNG instance URL")
    engines: List[str] = Field(default_factory=list)

    def __init__(self, **data: Any):
        """Initialize the SearxNG wrapper."""
        super().__init__(**data)
        # Import moved inside __init__ to prevent ModuleNotFoundError during linting
        try:
            from langchain_community.utilities import SearxSearchWrapper
            self._wrapper = SearxSearchWrapper(
                searx_host=self.host,
                engines=self.engines,
            )
        except ImportError:
            logger.error("langchain-community not installed. SearxNG will not work.")
            self._wrapper = None

    async def search(
        self, query: str, num_results: int = 5, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform search with error handling and result formatting."""
        if not self._wrapper:
            logger.error("SearxNG wrapper not initialized (missing dependencies)")
            return []

        try:
            # LangChain wrapper handles the API call
            raw_results = self._wrapper.results(
                query,
                num_results=num_results,
                **kwargs,
            )

            formatted_results = []
            for res in raw_results:
                formatted_results.append(
                    {
                        "title": res.get("title", ""),
                        "link": res.get("link", ""),
                        "snippet": res.get("snippet", ""),
                        "source": self.name,
                    }
                )
            return formatted_results

        except Exception as e:
            logger.error(f"SearxNG search failed: {str(e)}")
            return []
