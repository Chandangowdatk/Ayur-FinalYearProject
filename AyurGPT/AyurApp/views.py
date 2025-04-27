import time
import os
import json
import sqlite3
# import numpy as np
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import login
from knox.models import AuthToken
from groq import Groq
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from dotenv import load_dotenv
from pathlib import Path
from gtts import gTTS
import base64
import tempfile

from .serializers import UserSerializer, RegisterSerializer, LoginSerializer, ChatHistorySerializer
from .models import ChatHistory

# 📌 Constants
load_dotenv()
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "ayur_fixed_rag"

# Get the base directory and create absolute path for SQLite database
BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_DB_PATH = os.path.join(BASE_DIR, "L2_minilm_sentences.db")
print(f"Using SQLite database at: {SQLITE_DB_PATH}")

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # 🔹 Replace with your actual API key
# GROQ_API_KEY="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"
GROQ_API_KEY="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"

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ✅ Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
milvus_collection = Collection(name=MILVUS_COLLECTION)

# ✅ Connect to SQLite
conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()

# User Registration API
@api_view(['POST'])
@permission_classes([AllowAny])
def register_api(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        _, token = AuthToken.objects.create(user)
        
        return Response({
            "user": UserSerializer(user, context={"request": request}).data,
            "token": token
        })
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# User Login API
@api_view(['POST'])
@permission_classes([AllowAny])
def login_api(request):
    serializer = LoginSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.validated_data
        login(request, user)
        _, token = AuthToken.objects.create(user)
        
        return Response({
            "user": UserSerializer(user, context={"request": request}).data,
            "token": token
        })
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# Get User API
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_api(request):
    user = request.user
    return Response({
        "user": UserSerializer(user, context={"request": request}).data,
    })

# Get User Chat History
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chat_history(request):
    user = request.user
    chats = user.chats.all()
    serializer = ChatHistorySerializer(chats, many=True)
    return Response(serializer.data)

# Delete a Chat History Item
@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_chat_history(request, chat_id):
    user = request.user
    try:
        chat = ChatHistory.objects.get(id=chat_id, user=user)
        chat.delete()
        return Response({"message": "Chat history deleted successfully"}, status=204)
    except ChatHistory.DoesNotExist:
        return Response({"error": "Chat history not found or you don't have permission to delete it"}, status=404)

def query_similar_sanskrit(english_query, top_k=40):
    """Query Milvus for similar Sanskrit sentences using an English query."""
    
    # Open a new SQLite connection for this request
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    query_embedding = model.encode([english_query])[0].tolist()
    
    print(f"Querying Milvus with embedding of size: {len(query_embedding)}")

    search_results = milvus_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["sentence_id"]
    )
    
    print(f"Milvus returned {len(search_results[0])} results")

    results = []
    for hit in search_results[0]:
        sentence_id = hit.entity.get("sentence_id")
        distance = hit.distance
        print(f"Found hit with ID {sentence_id}, distance: {distance}")
        cursor.execute("SELECT full_text FROM sentences WHERE id = ?", (sentence_id,))
        result = cursor.fetchone()
        if result:
            results.append(result[0])
        else:
            print(f"No text found for sentence_id: {sentence_id}")

    # Close the connection to avoid thread issues
    conn.close()
    
    print(f"Retrieved {len(results)} Sanskrit sentences")
    return results


def generate_response(question, context_sentences):
    """Use Groq's LLaMA API to generate an answer based on the retrieved Sanskrit sentences."""
    context_text = "\n".join(context_sentences)
    prompt = (
        "You are an expert Ayurvedic doctor. Answer the question using the provided context.\n\n"
        "Guidelines:\n"
        "- Extract as much relevant information as possible from the context.\n"
        "- Only if you absolutely cannot find ANY relevant information, respond with 'I don't have enough information about this topic.'\n"
        "- Otherwise, do your best to answer with the available information.\n"
        "- Replace common medical terms with Ayurvedic terminology if appropriate.\n"
        "- Ensure the response aligns with Ayurveda's holistic approach.\n"
        "- If enough information is available, organize your answer into: Overview, Home Remedies, Dietary Recommendations, and Scientific Studies.\n"
        "- Be thorough and detailed in your response.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    print(f"Sending prompt of length {len(prompt)} to Groq")
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content.strip()

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def text_to_speech(request):
    """API endpoint to convert text to speech audio."""
    try:
        data = json.loads(request.body)
        text = data.get("text", "")
        language = data.get("language", "en")
        
        if not text:
            return JsonResponse({"error": "No text provided"}, status=400)
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            # Generate speech from text
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_audio.name)
            
            # Read the audio file as binary data
            with open(temp_audio.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Encode the binary data as base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Delete the temporary file
            os.unlink(temp_audio.name)
        
        return JsonResponse({
            "audio": audio_base64,
            "content_type": "audio/mp3"
        })
    
    except Exception as e:
        print(f"Error in text_to_speech endpoint: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def chat(request):
    """API endpoint to handle user queries."""
    try:
        data = json.loads(request.body)
        question = data.get("question", "")
        enable_tts = data.get("enable_tts", False)  # New parameter to enable text-to-speech
        print(f"Received question: {question}")
        
        if not question:
            return JsonResponse({"error": "No question provided"}, status=400)

        # Set environment variable to silence the tokenizer warnings
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        retrieved_sentences = query_similar_sanskrit(question, top_k=40)
        
        if retrieved_sentences:
            response = generate_response(question, retrieved_sentences)
        else:
            response = "I don't have enough information to answer this question."
        
        # Save the chat history for the user
        user = request.user
        ChatHistory.objects.create(
            user=user,
            question=question,
            answer=response
        )
        
        print(f"Number of retrieved sentences: {len(retrieved_sentences)}")
        print(f"First few retrieved sentences: {retrieved_sentences[:2]}")
        print(f"Response: {response[:100]}...")

        result = {"question": question, "answer": response}
        
        # Generate audio if text-to-speech is enabled
        # Only include audio in response if explicitly requested
        if enable_tts:
            try:
                # Create a temporary file to store the audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                    # Generate speech from text
                    tts = gTTS(text=response, lang="en", slow=False)
                    tts.save(temp_audio.name)
                    
                    # Read the audio file as binary data
                    with open(temp_audio.name, 'rb') as audio_file:
                        audio_data = audio_file.read()
                    
                    # Encode the binary data as base64
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # Delete the temporary file
                    os.unlink(temp_audio.name)
                
                result["audio"] = audio_base64
                result["content_type"] = "audio/mp3"
                
            except Exception as e:
                print(f"Error generating speech: {str(e)}")
                # Continue with response even if speech generation fails

        return JsonResponse(result)

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)