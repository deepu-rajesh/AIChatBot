import requests

google_api_key = ''
google_cx = ''

def google_search(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={google_cx}"
    response = requests.get(search_url)
    results = response.json()
    return results['items'][0]['snippet']
