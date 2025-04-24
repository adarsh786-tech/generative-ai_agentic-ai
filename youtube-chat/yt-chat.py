import re
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

load_dotenv()

def process_query(query, llmModel, system_prompt):
    messages = [
        ("system", system_prompt),
        ("human", query)
    ]
    return llmModel.invoke(messages).content

def extract_video_id(youtube_url):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def create_video_summary(all_chunks, llmModel, video_id):
    # Sample chunks from throughout the video
    total_chunks = len(all_chunks)
    if total_chunks > 50:
        # Take samples from beginning, middle, and end
        sampled_chunks = all_chunks[:10] + all_chunks[total_chunks//2-5:total_chunks//2+5] + all_chunks[-10:]
    else:
        sampled_chunks = all_chunks
    
    combined_content = "\n".join([f"[{chunk.metadata.get('timestamp')}] {chunk.page_content}" for chunk in sampled_chunks])
    
    messages = [
        ("system", "Create a comprehensive summary of this YouTube video transcript. Focus on the main topics, key points, and overall structure of the video."),
        ("human", combined_content)
    ]
    summary = llmModel.invoke(messages).content
    return summary

# Main video processing function
def setup_video_bot(video_url, llmModel):
    video_id = extract_video_id(video_url)
    print(f"Processing video ID: {video_id}")
    
    # Get transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    transcript_docs = []
    
    for entry in transcript:
        start_time = entry['start']
        text = entry['text']
        # Convert seconds to hh:mm:ss format
        minutes, seconds = divmod(start_time, 60)
        hours, minutes = divmod(minutes, 60)
        timestamp = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
        
        doc = Document(
            page_content=text,
            metadata={"timestamp": timestamp, "video_id": video_id}
        )
        transcript_docs.append(doc)
    
    # Create embeddings and vector store
    embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = QdrantVectorStore.from_documents(
        documents=transcript_docs,
        collection_name=f"yt-{video_id}",  # Use video-specific collection
        embedding=embedder,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    print("Transcript processed and stored in vector database")
    
    # Generate video summary
    video_summary = create_video_summary(transcript_docs, llmModel, video_id)
    print("Video summary generated")
    
    # Create full transcript text for reference
    full_transcript = "\n".join([f"[{doc.metadata.get('timestamp')}] {doc.page_content}" for doc in transcript_docs])
    
    return {
        "video_id": video_id,
        "vector_store": vector_store,
        "embedder": embedder,
        "transcript_docs": transcript_docs,
        "summary": video_summary,
        "full_transcript": full_transcript
    }

def get_response_for_query(query, video_data, llmModel):
    # Get relevant chunks based on the query
    retriever = QdrantVectorStore.from_existing_collection(
        collection_name=f"yt-{video_data['video_id']}",
        embedding=video_data['embedder'],
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Get more chunks for better context
    relevant_chunks = retriever.similarity_search(query=query, k=8)
    
    # Create system prompt with both summary and relevant chunks
    system_prompt = f"""
    You are a helpful AI Assistant who provides detailed information about YouTube videos.
    
    Video Summary:
    {video_data['summary']}
    
    Relevant Transcript Sections:
    {chr(10).join(f"[{chunk.metadata.get('timestamp', 'Unknown')}] {chunk.page_content}" for chunk in relevant_chunks)}
    
    General Instructions:
    - Answer the user's questions based on the transcript information
    - Reference timestamps when discussing specific parts of the video
    - If the user asks about something not covered in the relevant sections but might be in the video, indicate this
    - If asked for a summary, provide the comprehensive video summary
    - Your knowledge is limited to what's in this video transcript
    """
    
    # Special cases for certain types of queries
    if "summarize" in query.lower() or "summary" in query.lower():
        return f"Here's a summary of the video:\n\n{video_data['summary']}"
    
    return process_query(query, llmModel, system_prompt)

# Main execution
def main():
    # Initialize LLM
    llmModel = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    # Get video URL from user
    # video_url = input("Enter YouTube video URL: ").strip()
    # if not video_url:
    #     video_url = "https://www.youtube.com/watch?v=pNcQ5XXMgH4"  # Default video
    #     print(f"Using default video: {video_url}")
    video_url = "https://www.youtube.com/watch?v=pNcQ5XXMgH4"
    # Process video and set up bot
    video_data = setup_video_bot(video_url, llmModel)
    
    # Interactive chat loop
    print("\n" + "="*50)
    print(f"YouTube Video Bot Ready | Video ID: {video_data['video_id']}")
    print("Enter 'exit' or 'quit' to end the conversation")
    print("="*50 + "\n")
    
    while True:
        query = input("\nAsk about the video: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
            
        print("\nThinking...")
        response = get_response_for_query(query, video_data, llmModel)
        print("\nResponse:", response)
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()