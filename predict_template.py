from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings


# This is the schema for the prediction request.
# Note you do not need to use all the fields listed.
# For example if your predictor only needs data related to the content  
# then you can ignore the topic fields.
class TopicPredictionRequest(BaseModel):
    content_title: Optional[str] = None
    content_description: Optional[str] = None
    content_kind: Optional[str] = None
    content_text: Optional[str] = None
    topic_title: Optional[str] = None
    topic_description: Optional[str] = None
    topic_category: Optional[str] = None


class TopicPredictor:
    def __init__(self, token_file='token'):
        # Get token
        if token_file is not None:
            with open('token') as f:
                token = f.read()
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            raise Exception('HuggingFace API token not found')

        # Create embedding calculator
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            task="feature-extraction",
        )

        # Load topic vector store
        self.vector_store = Chroma(
            collection_name="topics",
            embedding_function=self.embeddings,
            persist_directory="data/vector_store",
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 5})

    def predict(self, request: TopicPredictionRequest) -> List[str]:
        """Takes in the request and can use all or some subset of input parameters to
        predict the topics associated with a given piece of content.

        Args:
            request (TopicPredictionRequest): See class TopicPredictionRequest

        Returns:
            List[str]: Should be list of topic ids.
        """
        all_results = None
        if request.content_title is not None:
            all_results = self.retrieve(f'What is topic of title: "{request.content_title}"?', 'title')
        if request.content_description is not None:
            new_df = self.retrieve(f'What is topic of description: "{request.content_description}"?', 'description')
            if all_results is not None:
                all_results = pd.concat([all_results, new_df])
        if request.content_text is not None:
            new_df = self.retrieve(f'What is topic of: "{request.content_text[:1000]}"?', 'text')
            if all_results is not None:
                all_results = pd.concat([all_results, new_df])
        all_results = all_results[['id', 'score']]
        all_results = all_results.groupby(['id'])['score'].sum()  # Grouping by rank
        all_results = all_results.sort_values(ascending=False)
        return all_results.iloc[:5].index.tolist() # Top 5 results

    def retrieve(self, prompt: str, type: str):
        """
        Helper function to retrieve a topic based on prompt and assign a specific type to it.
        :param prompt: str
        :param type: 
        :return: 
        """
        results = self.retriever.invoke(prompt)
        dicts = [r.metadata for r in results]
        for i, (r, d) in enumerate(zip(results, dicts)):
            d['score'] = 5 - i
            d['type'] = type
            d['name'] = r.page_content
        return pd.DataFrame(dicts)


if __name__ == '__main__':
    topics = pd.read_csv('data/topics.csv')
    content = pd.read_csv('data/content.csv', nrows=1000)
    random_content = content.sample()
    random_content.replace({np.nan: None}, inplace=True)
    request = TopicPredictionRequest(
        content_title=random_content['title'].item(),
        content_description=random_content['description'].item(),
        content_kind=random_content['kind'].item(),
        content_text=random_content['text'].item()
    )
    predictor = TopicPredictor()
    predicted = predictor.predict(request)
    topic_names = [topics[topics['id'] == id]['title'].item() for id in predicted]
    print('Content title:', random_content['title'].item())
    print('Predicted topics:', topic_names)
