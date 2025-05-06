import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Key (Replace with your actual Groq API key)
GROQ_API_KEY = ""  # Replace with your actual Groq API key

# Initialize Groq LLM
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",  # Or another Groq-supported model
        groq_api_key=GROQ_API_KEY,
        temperature=0.7,  # Adjust as needed
        max_tokens=500
    )
    logging.info("Groq LLM initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Groq LLM: {e}")
    llm = None

# Define the summarization prompt.
SUMMARY_PROMPT_TEMPLATE = """
Write a concise summary of the following text, no more than 200 words.
Include the main points and key details.

Text:
{text}
"""
SUMMARY_PROMPT = PromptTemplate(template=SUMMARY_PROMPT_TEMPLATE, input_variables=["text"])


def extract_article_content(article_url):
    """
    Fetches and extracts the main content from a single news article page.

    Args:
        article_url (str): The URL of the news article.

    Returns:
        str: The extracted text content of the article, or an empty string if not found.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        print(f"Fetching article content from: {article_url}")
        response = requests.get(article_url, headers=headers, timeout=100)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # --- IMPORTANT: SELECTORS FOR ARTICLE BODY ---
        # These selectors are common for article content. You WILL LIKELY need to
        # inspect Moneycontrol article pages and ADJUST these selectors.
        # Add more selectors if needed, or make them more specific.
        article_body_selectors = [
            'div.content_wrapper',              # Often used for the main content area
            'div.article_content',              # Common class for article text
            'div#contentdata',                  # Sometimes content is within an ID like this
            'div.art_content',                  # Another variation
            'div.story_page',                   # Specific to some news layouts
            'div.text',                         # General class that might contain article text
            'article .post-content',            # HTML5 <article> tag
            'div[class*="article-body"]',       # Catches classes like "article-body-class"
            'div.main-content-body',             # Another common pattern
            'div.content_detail__body',          # Moneycontrol specific pattern seen
            'div.content_text',                  # Another Moneycontrol specific pattern
            'div.artText'                       # Short for article text
        ]

        article_text = ""
        for selector in article_body_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                # Extract text, trying to remove script/style tags and excessive newlines
                for unwanted_tag in content_element.find_all(['script', 'style', 'aside', '.related_stories', '.embed-container']):
                    unwanted_tag.decompose()
                
                # Get text from all relevant child elements, joining with spaces
                text_parts = [text.strip() for text in content_element.stripped_strings]
                article_text = "\n".join(filter(None, text_parts)) # Join non-empty lines
                
                if article_text.strip(): # If we found substantial text
                    print(f"Successfully extracted content using selector: '{selector}'")
                    break # Stop if content is found
            else:
                print(f"Selector '{selector}' did not find any content on {article_url}")


        if not article_text.strip():
            print(f"Could not extract main article content from {article_url} with current selectors.")
            # As a last resort, try to get all paragraph tags if no specific container was found
            paragraphs = soup.find_all('p')
            if paragraphs:
                article_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                if article_text.strip():
                     print("Extracted content using generic paragraph search as a fallback.")
                else:
                    print("Generic paragraph search also yielded no significant content.")
            else:
                print("No paragraph tags found for fallback extraction.")


        return article_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching article {article_url}: {e}")
    except Exception as e:
        print(f"An error occurred while extracting content from {article_url}: {e}")
    return ""

# def fetch_news_article(url):
#     """
#     Fetches the text content of a news article from a given URL.

#     Args:
#         url (str): The URL of the news article.

#     Returns:
#         str: The text content of the article, or None on error.
#     """
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise exception for bad status
#         soup = BeautifulSoup(response.text, 'html.parser')
#         article_content = soup.find('div', class_='article-content')  # Example
#         print(article_content)
#         if article_content:
#             paragraphs = article_content.find_all('p')
#             text = ' '.join([p.text.strip() for p in paragraphs])
#             return text
#         else:
#             logging.warning(f"Could not extract article content from {url}. Selector may be incorrect.")
#             return None
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error fetching article from {url}: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Error processing article from {url}: {e}")
#         return None



def summarize_text(article_text):
    """
    Summarizes the given text using the Groq model.

    Args:
        article_text (str): The text to summarize.

    Returns:
        str: The summary, or None on error.
    """
    if llm is None:
        logging.error("LLM is not initialized. Cannot summarize.")
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        texts = text_splitter.create_documents([article_text])
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=SUMMARY_PROMPT)
        summary = chain.invoke(texts)
        return summary
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return None

def get_news_summary(article_url):
    """
    Fetches a news article from the given URL and generates a summary.

    Args:
        article_url (str): The URL of the news article.

    Returns:
        str: The summary of the article, or None on error.
    """
    article_text = extract_article_content(article_url)

    if article_text:
        summary = summarize_text(article_text)
        if summary:
            return summary
        else:
            return "Failed to generate summary."
    else:
        return "Failed to fetch article content."


def fetch_news_based_on_preferences(preferences):
    """
    Fetches news articles based on user preferences (keywords) and summarizes them.
    This is a placeholder, as actual news fetching requires a news API or scraping.

    Args:
        preferences (list): A list of keywords representing user preferences.

    Returns:
        dict: A dictionary where keys are article titles and values are summaries.
              Returns an empty dict if no articles are found or on error.
    """
    # Placeholder
    example_articles = [
        {
            "title": " Sharechat swaps cash burn for sustainable growth, eyeing IPO in 2 yrs: CEO Ankush Sachdeva",
            "article_url": "https://www.moneycontrol.com/technology/sharechat-swaps-cash-burn-for-sustainable-growth-eyeing-ipo-in-2-yrs-ceo-ankush-sachdeva-article-13010134.html",
        },
        # {
        #     "title": "New Study on Climate Change",
        #     "url": "https://www.example.com/climate-change-study",
        # },
        # {
        #     "title": "Stock Market Hits New Record",
        #     "url": "https://www.example.com/stock-market-record",
        # },
        #  {
        #     "title": "Local School Wins National Award",
        #     "url": "https://www.example.com/local-school-award"
        # },
        # {
        #     "title": "New Electric Vehicle Launched",
        #     "url": "https://www.example.com/ev-launch"
        # }
    ]

    # Filter articles based on preferences
    relevant_articles = [
        article for article in example_articles
        if any(keyword.lower() in article["title"].lower() for keyword in preferences)
    ]

    if not relevant_articles:
        logging.info(f"No articles found matching preferences: {preferences}")
        return {}

    results = {}

    for article in relevant_articles:
        summary = get_news_summary(article["article_url"])
        if summary:
            results[article["title"]] = summary
        else:
            results[article["title"]] = "Failed to retrieve or summarize article."
    return results

if __name__ == "__main__":
    user_preferences = ["AI", "Climate Change", "Stock Market"]
    news_summaries = fetch_news_based_on_preferences(user_preferences)

    if news_summaries:
        print("News Summaries based on your preferences:")
        for title, summary in news_summaries.items():
            print(f"\nTitle: {title}")
            print(f"Summary: {summary['output_text']}")
    else:
        print(f"No news found for your preferences: {user_preferences}")
