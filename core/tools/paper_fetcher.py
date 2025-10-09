import logging
import arxiv
from typing import List, Dict

def search_papers(query: str) -> List[Dict]:
    try:
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results())
        if not results:
            return []
        
        paper_list = []
        for res in results:
            paper_list.append({
                "id": res.entry_id.split('/')[-1],
                "title": res.title,
                "summary": res.summary,
                "authors": [author.name for author in res.authors],
                "url": res.pdf_url,
                "status": "pending"
            })
        return paper_list
    except Exception as e:
        logging.error(f"Ошибка при поиске статей: {e}")
        return []