from googlesearch import search
import requests
from bs4 import BeautifulSoup

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import cohere

#Semantic chunking model details
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
)


def get_top_10_urls(user_query, k = 2):
    urls = []
    try:
        # Perform the search and retrieve the first 10 results
        for url in search(user_query):
            urls.append(url)
            if len(urls) >= k:  # Stop once we have 10 URLs
                break
    except Exception as e:
        print(f"Error occurred during Google search: {e}")

    return urls


def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)  # Send a GET request with a 10-second timeout
        if response.status_code == 200:  # Check if the request was successful
            soup = BeautifulSoup(response.text, 'html.parser')  # Parse the HTML content
            text = soup.get_text(separator=' ', strip=True)  # Extract the text
            return text
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def return_documents_splitters(text):

    text_splitter = SemanticChunker(hf, breakpoint_threshold_type="percentile")
    docs = text_splitter.create_documents([text])
    return docs


def get_texts_from_urls(urls):
    texts = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            texts.append(text)

    collection = []
    # for text in texts:
    #     collection.extend(return_documents_splitters(text))
    # print(">>Semantic Chunking")
    return texts


def get_output_faiss(texts, user_query):
    from langchain_huggingface import HuggingFaceEmbeddings
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from uuid import uuid4
    from langchain_core.documents import Document


    # embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

    index = faiss.IndexFlatL2(len(hf.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=hf,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    docs = list()
    for text in texts:
        docs.append(Document(page_content=text, metadata={"source": "google search"}))

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    context = [i.page_content for i in retriever.invoke(user_query)]

    COHERE_API_KEY_TEXT = "QiQxhenWCqbLXCmTGSRL4O9g0uNeVhxraSWwzAfO"

    # if len(history)<3:
    prompt = f"""
        You are a chatbot built for helping users queries on their diet.
        The user wants to lead a healthy life style and healthy diet.
        Just Answer only about the side effects of the product in human body. Do not suggest anything.
        Answer the question:{user_query} only based on the context: {context} provided.
        Try to answer as a paragraph.
        Give easy to understand responses.
        Do not divulge any other details other than query or context.
        If the question asked is a generic question or causal question answer them without using the context.
        If the question is a general question, try to interact with the user in a polite way. """
    # else:
    #     prompt = f"""
    #     You are a chatbot built for helping users queries.
    #     Answer the question:{user_query} only based on the context: {context} chat history:{history[len(history)-3:]} provided.
    #     Try to answer in bulletin points.
    #     Give easy to understand responses.
    #     Do not divulge any other details other than query or context.
    #     If the question asked is a generic question or causal question answer them without using the context.
    #     If the question is a general question, try to interact with the user in a polite way. """

    co = cohere.Client(COHERE_API_KEY_TEXT)
    response = co.chat(message=prompt, model="command-r", temperature=0)
    # history.append(response.text)
    return response.text


def rag_pipeline(user_query, claim, ingredients, age, sex, height, weight, activity):
    import json
    from web_scraper.nutrient_scraper import extractor, get_base_url, get_nutrients
    from image_extractor.services import get_harmful_ingredient
    from web_scraper.analyse_claim import analyze_claim
    nutrients_needed, nutrients_range = get_nutrients(extractor(get_base_url(age, sex, height, weight, activity)))
    harmful_ingredient = get_harmful_ingredient(ingredients)
    response = analyze_claim(claim, ingredients)
    response = json.loads(response)

    # print(nutrients_needed)
    # print(nutrients_range)
    # print(harmful_ingredient)
    # print(response)


    query = f"{harmful_ingredient} side effects"
    top_urls = get_top_10_urls(query, k = 10)
    texts = get_texts_from_urls(top_urls)

    # faiss_output = get_output_faiss(texts, query)

    # print(faiss_output)

    context = f"""
        The user wants to eat a snack which has the following claim: {claim}.
        The ingredients in that snack are: {ingredients}.
        The user's dietrician prescribed him the following diet: {nutrients_needed} and {nutrients_range}.
        The detailed analysis of the snack according to the above claim and ingredients is: {response}.
        According to leading websites, the most harmful ingredient in the snack is: {harmful_ingredient}.
        Also tell the user about other side effects using the ingredients present in it.
    """

    COHERE_API_KEY_TEXT = "QiQxhenWCqbLXCmTGSRL4O9g0uNeVhxraSWwzAfO"

    # if len(history)<3:
    prompt = f"""
        You are a chatbot built for helping users queries on their diet.
        The user wants to lead a healthy life style and healthy diet.
        Answer the question:{user_query} only based on the context: {context} provided.
        Suggest the user whether to eat this snack or not. If the user should not eat this, suggest any other healthy snacks.
        Try to answer as bulletin points.
        Give easy to understand responses.
        Do not divulge any other details other than query or context.
        If the question asked is a generic question or causal question answer them without using the context.
        If the question is a general question, try to interact with the user in a polite way. """
    # else:
    #     prompt = f"""
    #     You are a chatbot built for helping users queries.
    #     Answer the question:{user_query} only based on the context: {context} chat history:{history[len(history)-3:]} provided.
    #     Try to answer in bulletin points.
    #     Give easy to understand responses.
    #     Do not divulge any other details other than query or context.
    #     If the question asked is a generic question or causal question answer them without using the context.
    #     If the question is a general question, try to interact with the user in a polite way. """

    co = cohere.Client(COHERE_API_KEY_TEXT)
    response = co.chat(message=prompt, model="command-r", temperature=0)
    # history.append(response.text)

    print("\n\n")
    return response.text

