import os
import uuid
import ffmpeg
import onnxruntime as ort
from transformers import AutoTokenizer, pipeline
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, SearchParams

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from typing import List, Optional


def extract_audio(video_path, output_audio_path):
    """
    Extract the audio from a given video file, and save it to a specified file path.

    The extracted audio will be a single-channel, 16-bit PCM WAV file, with a sample
    rate of 16,000 Hz.

    Parameters
    ----------
    video_path : str
        The path to the video file from which to extract the audio
    output_audio_path : str
        The path to the file to which to save the extracted audio
    """
    (
        ffmpeg
        .input(video_path)
        .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run()
    )

def transcribe_audio(audio_path):
    
    """
    Transcribes the given audio file into text.

    Parameters
    ----------
    audio_path : str
        The path to the audio file to transcribe

    Returns
    -------
    str
        The text transcription of the audio file
    """

    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small", return_timestamps=True) 
    result = transcriber(audio_path)
    return result["text"]


def transcribe_video(video_path: str) -> str:

    """
    Transcribes the audio from a given video file into text.

    This function extracts the audio from the specified video file, saves it as a temporary audio file, 
    and then uses an automatic speech recognition model to convert the audio into text. 
    The transcribed text is returned as output.

    Args:
        video_path (str): The path to the video file to be transcribed.

    Returns:
        str: The transcribed text from the audio of the video.
    """



    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    text = transcribe_audio(audio_path)
    return text

def create_client_and_collection(collection_name: str = "makeathon") -> QdrantClient:

    """
    Creates a QdrantClient instance and a collection with the specified name.

    This function sets up a QdrantClient instance that connects to a local Qdrant server
    (host="localhost", port=6333) and creates a collection with the specified name or
    recreates it if it already exists. The collection is configured to store vectors
    of size 1024 with cosine distance.

    Args:
        collection_name (str): The name of the collection to be created (default: "makeathon").

    Returns:
        QdrantClient: The instance of the QdrantClient.
    """
    client = QdrantClient(host="localhost", port=6333)

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE) 
    )

    return client

def index_data(client: QdrantClient, filepath:str, filename: str, ort_session: ort.InferenceSession, 
               tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3"), collection_name: str = "makeathon") -> None:

    """
    Indexes the content of a file in a Qdrant collection.

    This function takes a file path and filename, an ONNX runtime session, and an optional tokenizer.
    It supports PDF, MP4, and MP3 files. It calls the appropriate partitioning function to extract
    meaningful chunks of text from the file. It then chunks the text by title using the
    `chunk_by_title` function from `unstructured.partition.text`. It then embeds each chunk of text
    using the provided tokenizer and the ONNX runtime session, and stores the embeddings in a Qdrant
    collection with the specified name.

    Args:
        client (QdrantClient): The instance of the QdrantClient.
        filepath (str): The path to the file to be indexed.
        filename (str): The name of the file to be indexed.
        ort_session (ort.InferenceSession): The ONNX runtime session to be used for embeddings.
        tokenizer (AutoTokenizer): The tokenizer to be used for embeddings (default: BAAI/bge-m3).
        collection_name (str): The name of the Qdrant collection to be used (default: "makeathon").

    Returns:
        None
    """

    if filepath.endswith(".pdf"):
        elements  = partition_pdf(filename=filepath, ocr_strategy="auto")
    elif filepath.endswith(".mp4"):
        transcription = transcribe_video(filepath)
        elements = partition_text(text = transcription)
    elif filepath.endswith(".mp3"):
        transcription = transcribe_audio(filepath)
        elements = partition_text(text = transcription)
    else:
        raise ValueError(f"Unsupported file format")
    
    chunks = chunk_by_title(elements)

    for chunk in chunks:
        text = chunk.text
        inputs = tokenizer(text, padding="longest", return_tensors="np")
        inputs_onnx = {k: v for k, v in inputs.items()}
        embedding = ort_session.run(None, inputs_onnx)
        
        point_id = str(uuid.uuid4())  # Generate a random UUID
        point = PointStruct(
            id=point_id,
            vector=embedding[0][0],
            payload={"filename": filename ,"original_text": text}
        )

        client.upsert(
            collection_name=collection_name,
            points=[point]
        )

def retrieve_text( query: str, client: QdrantClient, ort_session: ort.InferenceSession, 
                  tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3"), collection_name: str = "makeathon", limit: int = 20, filename: str = None) -> Optional[List[str]]:

    """
    Retrieves text content from a Qdrant collection based on a query and optional filename filter.

    This function performs a search in the specified Qdrant collection using a query embedding, 
    and optionally filters the results by filename. It returns a list of original text contents 
    from the search results.

    Args:
        query (str): The search query to be tokenized and embedded.
        client (QdrantClient): The instance of the QdrantClient used for querying.
        ort_session (ort.InferenceSession): The ONNX session used for generating embeddings.
        tokenizer (AutoTokenizer): The tokenizer used to process the query.
        collection_name (str): The name of the Qdrant collection to search in (default: "makeathon").
        limit (int): The maximum number of search results to return (default: 20).
        filename (str, optional): The filename to filter the search results by.

    Returns:
        Optional[List[str]]: A list of original text contents from the search results, or an empty list if no results are found.
    """

    if query:
        inputs = tokenizer(query, padding="longest", return_tensors="np")
        inputs_onnx = {k: v for k, v in inputs.items()}
        query_embedding = ort_session.run(None, inputs_onnx)[0]

    if filename:

        payload_filter = Filter(
        must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename)
                )
            ]
        )

        if query:

            search_result = client.query_points(
                collection_name="makeathon",
                query=query_embedding[0],
                query_filter=payload_filter,
                limit=10,  # Number of results
                search_params=SearchParams(hnsw_ef=128)  # Optional tuning
            )

        else:

            search_result = client.query_points(
                collection_name="makeathon",
                query_filter=payload_filter,
                limit=10,  # Number of results
                search_params=SearchParams(hnsw_ef=128)  # Optional tuning
            )

    else:
        # Assuming `query_embedding` shape is (1, dim)
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_embedding[0],  # extract the 1D vector
            limit=limit,  
        )

    try:
        text_list =  [result[1][0].payload['original_text'] for result in search_result]
        return text_list
    except:
        return []

