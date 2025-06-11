import os
import json
import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.config import client, groq_client, langfuse
from app.services.cost_service import calculate_cost
from app.services.qdrant_service import search_scenarios_in_qdrant_async
from langfuse import Langfuse

router = APIRouter()

def get_recent_messages(messages):
    assistant_contents = [None, None]  # [second_to_last, last]
    user_contents = [None, None]      # [second_to_last, last]
    
    for message in reversed(messages):
        role = message.get('role')
        if role in ('tool', 'function') or message.get('tool_call'):
            continue
        content = message.get('content')
        
        if role == 'assistant':
            if assistant_contents[1] is None:
                assistant_contents[1] = content
            elif assistant_contents[0] is None:
                assistant_contents[0] = content
        elif role == 'user':
            if user_contents[1] is None:
                user_contents[1] = content
            elif user_contents[0] is None:
                user_contents[0] = content
                
        if all(assistant_contents) and all(user_contents):
            break
    
    # Format the output in chronological order
    return f"Assistant: {assistant_contents[0] or 'N/A'}\nUser: {user_contents[0] or 'N/A'}\nAssistant: {assistant_contents[1] or 'N/A'}\nUser: {user_contents[1] or 'N/A'}"

def system_prompt_inject(trieve_response, messages):
    if not messages or messages[0].get('role') != 'system':
        return messages
    messages[0]['content'] += f"\n\nRelevant context+guidelines:\n{trieve_response}"
    return messages

@router.post("/chat/completions")
async def chat_proxy(request: Request):
    try:
        start_time = time.time()
        payload = await request.json()
        print(f"Raw Payload: {payload}")
        session_id = payload.get('call', {}).get('id', 'unknown_session') # Provide default for seed safety
        assistant_name = payload.get('assistant', 'unknown_assistant')

        # ---------------- Langfuse root span (v3) - WRAPPING THE ENTIRE LOGIC ----------------
        deterministic_trace_id = Langfuse.create_trace_id(seed=session_id or 'unknown_session')

        with langfuse.start_as_current_span(
            name=f"chat-proxy-request-{session_id}", # Name for the trace's root span
            trace_context={"trace_id": deterministic_trace_id}, # Link to your deterministic trace ID
            input={"original_request_payload": payload}, # Log the initial request as overall trace input
        ) as root_span:
            # Attach high-level trace attributes so that all nested operations
            # (RAG search, LLM generations, etc.) are grouped correctly.
            root_span.update_trace(
                user_id=session_id,
                session_id=session_id,
                tags=[str(assistant_name), "chat_flow"],
                metadata={"client_metadata": payload.get('metadata')}, # Pass through any other top-level metadata
            )

            keys_to_remove = ['call', 'metadata', 'activeAssistant', 'credentials', 'toolDefinitionsExcluded', 'customer', 'phoneNumber', 'assistant', 'timestamp']
            for key in keys_to_remove:
                if key in payload:
                    del payload[key]
        
            # --- RAG Search Span: Wrap it in its own span ---
            # This will be a child of the `root_span`.
            with langfuse.start_as_current_span(name="rag-search") as rag_span:
                rag_query = get_recent_messages(payload.get('messages', []))
                rag_span.update(input={"query": rag_query}) # Log the RAG input

                rag_search_start_time = time.time()
                qdrant_results = []
                current_qdrant_client = request.app.state.qdrant_client
                current_embedding_model = request.app.state.embedding_model

                if current_qdrant_client and current_embedding_model:
                    qdrant_results = await search_scenarios_in_qdrant_async(
                        qdrant_client_instance=current_qdrant_client,
                        embedding_model_instance=current_embedding_model,
                        query_text=rag_query,
                        top_k=5
                    )
                else:
                    print("Qdrant client or embedding model not initialized from app.state. Skipping Qdrant search in /chat/completions.")
                    rag_span.update(level="WARNING", status_message="Qdrant not initialized, search skipped.")

                formatted_rag_results = []
                if qdrant_results:
                    for hit in qdrant_results:
                        context = hit.get("payload", {}).get("context", "").strip()
                        guidelines = hit.get("payload", {}).get("responseGuidelines", "").strip()
                        if context or guidelines:
                            formatted_rag_results.append(f"- {context} {guidelines}".strip())

                rag_response_string = "\n".join(formatted_rag_results)
                if not rag_response_string:
                    rag_response_string = "No relevant context found from knowledge base."

                rag_search_speed = time.time() - rag_search_start_time
                print(f"Total RAG Time: {rag_search_speed:.3f} seconds")
                rag_span.update(
                    output={"rag_results": rag_response_string},
                    metadata={"duration_seconds": rag_search_speed}
                )
            # 'rag-search' span ends here.

            payload['messages'] = system_prompt_inject(rag_response_string, payload.get('messages', []))
            payload['stream_options'] = {"include_usage": True}

            print(f"Payload sent to LLM: {payload}")

            model_name = payload.get("model", "default")

            # --- LLM Call (Generation) ---
            # These calls will automatically be traced as Langfuse Generations
            # and nested under the 'root_span' because it's the current active context.
            llm_request_start_time = time.time() # This is the start time of the LLM call

            if model_name == "llama-3.3-70b-versatile":
                stream = await groq_client.chat.completions.create(**payload)
            else:
                stream = await client.chat.completions.create(**payload)

            async def logging_event_stream():
                # The `nonlocal start_time` refers to the `start_time` outside `chat_proxy`,
                # which is the start of the *entire request*.
                # For TTFT, you'd want the start time of the LLM request specifically.
                # We'll use `llm_request_start_time` for TTFT.
                _ttft_logged = False
                last_chunk = None
                collected_response_text = []

                # The `langfuse.openai` wrapper already creates the generation.
                # We can get a reference to it to add custom metadata/updates
                # if needed, but much is handled automatically.
                current_llm_generation = langfuse.get_current_observation()

                try:
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            delta_content = chunk.choices[0].delta.content
                            collected_response_text.append(delta_content)

                            if not _ttft_logged:
                                ttft = time.time() - llm_request_start_time # Use LLM request start time for TTFT
                                print(f"TTFT: {ttft:.3f} seconds")
                                if current_llm_generation:
                                    current_llm_generation.update(metadata={"time_to_first_token": ttft})
                                _ttft_logged = True

                        last_chunk = chunk
                        json_data = chunk.model_dump_json()
                        yield f"data: {json_data}\n\n"

                    final_response = "".join(collected_response_text)
                    print(f"Final LLM Response: {final_response}")

                    # Langfuse OpenAI wrapper typically handles final output and usage.
                    # We explicitly add cost here as it's a custom calculation.
                    if current_llm_generation:
                        prompt_tokens = 0
                        completion_tokens = 0
                        if last_chunk and hasattr(last_chunk, "usage") and last_chunk.usage:
                            prompt_tokens = getattr(last_chunk.usage, "prompt_tokens", 0)
                            completion_tokens = getattr(last_chunk.usage, "completion_tokens", 0)
                        else: # Fallback for token estimation
                            prompt_text = "".join([msg.get('content', '') for msg in payload.get('messages', []) if isinstance(msg, dict) and msg.get('content')])
                            prompt_tokens = len(prompt_text) // 4 + 1
                            completion_tokens = len(final_response) // 4 + 1
                            current_llm_generation.update(level="WARNING", status_message="Usage details missing from LLM response, tokens estimated.")

                        cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
                        current_llm_generation.update(cost_details={"total_cost": cost})
                        print(f"Cost: ${cost:.6f}")

                    # Update the overall trace output (on the root_span's trace)
                    root_span.update_trace(output={"final_llm_response": final_response})

                    total_request_time = time.time() - start_time
                    print(f"Total Request Processing Time: {total_request_time:.3f} seconds")

                except Exception as ex_stream:
                    print(f"Error during stream processing: {ex_stream}")
                    partial_response_text = "".join(collected_response_text)
                    print(f"Partial Response: {partial_response_text}")

                    # Mark the current LLM generation (if available) as ERROR
                    if current_llm_generation:
                        current_llm_generation.update(
                            level="ERROR",
                            status_message=f"Stream processing error: {ex_stream}",
                            output=partial_response_text
                        )

                    # Mark the overall trace as ERROR
                    root_span.update_trace(
                        level="ERROR",
                        status_message=f"Overall request failed due to stream error: {ex_stream}",
                        output={"partial_response": partial_response_text, "error": str(ex_stream)}
                    )

                    # Re-calculate cost based on partial response if needed
                    prompt_text = "".join([msg.get('content', '') for msg in payload.get('messages', []) if isinstance(msg, dict) and msg.get('content')])
                    estimated_prompt_tokens = len(prompt_text) // 4 + 1
                    estimated_completion_tokens = len(partial_response_text) // 4 + 1
                    cost = calculate_cost(model_name, estimated_prompt_tokens, estimated_completion_tokens)
                    if current_llm_generation:
                        current_llm_generation.update(cost_details={"total_cost": cost})
                    print(f"Cost (on error): ${cost:.6f}")

                    total_request_time = time.time() - start_time
                    print(f"Total Request Processing Time (on error): {total_request_time:.3f} seconds")

            return StreamingResponse(logging_event_stream(), media_type="text/event-stream")

        # --- END: Your existing business logic. The root_span automatically ends here. ---

    except Exception as e:
        import traceback
        print(f"Error in /chat/completions endpoint: {str(e)}\nTraceback: {traceback.format_exc()}")

        # Fallback for very early errors outside any established trace context
        # This creates a separate 'event' trace for this specific error.
        try:
            payload_for_error = payload
        except NameError:
            payload_for_error = "Payload not available"
            
        try:
            session_id_for_error = session_id
        except NameError:
            session_id_for_error = "unknown_session"

        langfuse.start_event(
            name="chat_proxy_endpoint_error",
            level="ERROR",
            input={"initial_payload_on_error": payload_for_error}, # Log the payload that caused the early error
            status_message=f"Top-level endpoint error before trace established: {e}",
            metadata={"traceback": traceback.format_exc(), "error_type": type(e).__name__, "session_id": session_id_for_error}
        ).end() # IMPORTANT: Manually end this event as it's not a context manager

        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") 