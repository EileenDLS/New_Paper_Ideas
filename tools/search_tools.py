import json
import os

import requests
from langchain.tools import tool
import arxiv


class SearchTools():

  @tool("Search internet")
  def search_internet(query):
    """Useful to search the internet about a given topic and return relevant
    results."""
    return SearchTools.search_google_api(query)
  
  @tool("Search Arxiv")
  def search_arxiv(query):
    """Useful to search for academic papers on Arxiv about a given topic and return relevant results."""
    query = f"site:arxiv.org {query}"
    return SearchTools.search_arxiv_api(query)


  def search_google_api(query, n_results=5):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['organic']
    stirng = []
    for result in results[:n_results]:
      try:
        stirng.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    content = '\n'.join(stirng)
    return f"\nSearch result: {content}\n"

  def search_arxiv_api(query, max_results=5):
    search = arxiv.Search(
      query=query,
      max_results=max_results,
      sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for result in search.results():
      results.append({
        'title': result.title,
        'authors': ', '.join([author.name for author in result.authors]),
        'published': result.published.strftime('%Y-%m-%d'),
        'summary': result.summary,
        'url': result.entry_id
      })
    content = '\n-----------------\n'.join(
      [f"Title: {r['title']}\nAuthors: {r['authors']}\nPublished: {r['published']}\nSummary: {r['summary']}\nLink: {r['url']}" for r in results]
    )
    return f"\nArxiv search result: {content}\n"