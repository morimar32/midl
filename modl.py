from typing import List
from pydantic import BaseModel
from llama_cpp import Llama

# Global model instance
llm = None

class ChatMessageInput(BaseModel):
    role: str
    content: str

def initialize(config: dict) -> None:
    """Initialize the LLM model with configuration."""
    global llm
    llm = Llama(
        model_path=config['model_path'],
        n_ctx=config['n_ctx'],
        n_gpu_layers=config['n_gpu_layers']
    )

def process_request(messages: List[ChatMessageInput], config: dict = None) -> str:
    """Process a chat request using the initialized model."""
    if llm is None:
        raise RuntimeError("Model not initialized. Call initialize() first.") 

    # Format messages for the model
    formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages[:-1]]
    enriched_content = enrich_request(messages[-1].content, config)
    formatted_messages.append({"role": messages[-1].role, "content": enriched_content})
    
    # Get response from model
    response = llm.create_chat_completion(
        messages=formatted_messages,
        temperature=config.get('temperature', 0.6) if config else 0.6,
        top_p=config.get('top_p', 0.95) if config else 0.95,
        max_tokens=config.get('max_tokens', 32768) if config else 32768,
    )
    
    return response["choices"][0]["message"]["content"]

def enrich_request(latest_message: str, config: dict = None) -> str:
    """
    Enhance the user's request by generating a refined prompt using the LLM.
    
    Args:
        latest_message: The most recent user message to enrich
        config: Optional configuration for the LLM call
        
    Returns:
        The refined prompt extracted from the LLM response between &lt;refined_prompt&gt; tags
    """
    if llm is None:
        raise RuntimeError("Model not initialized. Call initialize() first.")
    
    # Internal prompt template to refine the user's request
    prompt_template = """<prompt>
    <greeting>Dear Language Model,</greeting>
    <instruction>
        <task>Please carefully follow these steps to reflect upon and enrich a given sample prompt. **Ensure that your reflection and intermediate thoughts during this process are enclosed within `<think>` XML tags in your response. The final enriched prompt should be placed outside of these tags.**</task>
        <sub_task order="1">
            <title>Understand the Core Request</title>
            <description>First, take a moment to **fully read and deeply understand the central question or request** presented within the `<sample_prompt>` tag below. Identify the user's primary goal or information need.</description>
        </sub_task>
        <sub_task order="2">
            <title>Reflect on Underlying Assumptions and Ambiguities</title>
            <description>Next, **critically reflect on any underlying assumptions** that might be present in the sample prompt. Consider potential **ambiguities or different interpretations** the prompt could have. Think about what might be unclear or left unsaid.</description>
        </sub_task>
        <sub_task order="3">
            <title>Brainstorm Related Concepts and Ideas</title>
            <description>Now, **generate related concepts and ideas** that naturally extend from the core of the sample prompt. Think broadly and consider different facets or aspects that connect to the original request. What related areas might the user be interested in or might be important to consider?</description>
        </sub_task>
        <sub_task order="4">
            <title>Identify Areas for Enrichment and Detail</title>
            <description>**Pinpoint specific areas where the sample prompt could be enriched** with more detail, explanation, or context to elicit a more comprehensive and insightful response. Consider what additional information or perspective would be beneficial.</description>
        </sub_task>
        <sub_task order="5">
            <title>Formulate Elaborations and Follow-Up Questions</title>
            <description>**Develop specific elaborations or insightful follow-up questions** that could help to create a more complete and nuanced understanding of the topic initiated by the sample prompt. These should aim to address the ambiguities identified and explore the related concepts brainstormed.</description>
        </sub_task>
    </instruction>
    <reflection_guidance>
        <elaboration_help>To ensure your elaborations are broadly helpful and useful for enriching a wide range of prompts, consider the following aspects during your reflection (please include these considerations within your `<think>` block):</elaboration_help>
        <ul>
            <li>**Specificity:** Does the original prompt ask for something specific, or could it be narrowed down for a more focused answer? Think about including specific types of information, timeframes, locations, or criteria.</li>
            <li>**Granularity:** Does the prompt ask for a high-level overview, or would a more detailed, fine-grained response be more valuable? Consider asking for mechanisms, processes, or specific examples.</li>
            <li>**Context:** Is there any missing context that would help in understanding the user's underlying need? Think about the user's background, their intended use of the information, or the broader situation.</li>
            <li>**Assumptions:** What assumptions might the original prompt be making? Are these assumptions valid, or should they be questioned or explored?</li>
            <li>**Scope:** Is the scope of the prompt too broad or too narrow? Could it be adjusted to yield more manageable or more comprehensive results?</li>
            <li>**Format:** Does the prompt imply a desired output format (e.g., a list, a comparison)? Could explicitly specifying a format improve the clarity or usefulness of the response?</li>
            <li>**Perspective:** Would considering different perspectives or viewpoints enrich the understanding of the topic? Think about different stakeholders or roles.</li>
            <li>**Underlying Goals:** What is the user really trying to achieve by asking this question? Understanding the underlying goal can help in tailoring the enriched prompt.</li>
        </ul>
    </reflection_guidance>
    <hallucination_warning>
        **Critical Warning:** Under no circumstances should you invent or hallucinate information that is not directly derived from a logical reflection on the provided sample prompt and the principles of effective prompt engineering. Your goal is to *enrich* the prompt, not to answer it or introduce external knowledge.
    </hallucination_warning>
    <reward_system>
    **You will be recognized for:**
      <reward>You will be considered to have provided a high-quality output if your refined prompt is significantly clearer, more specific, and more likely to elicit a detailed and insightful response compared to the original sample prompt. Your thoughtful reflection and the relevance of your elaborations will be highly valued.</reward>
</reward>
      <reward>Ensuring your response is correctly formatted within the `<think>` and `<refined_prompt>` XML structure.</reward>
    **Accuracy and attention to detail will be highly rewarded in our assessment.**
    </reward_system>
    <sample_prompt>What are the main factors contributing to the success of a small local bookstore?</sample_prompt>
    <refined_prompt>
        <!-- The language model will generate the refined prompt here based on the above instructions, outside of the <think> tags -->
    </refined_prompt>
    <closing>Thank you for your thoughtful work!</closing>
</prompt>

BELOW IS THE SAMPLE PROMPT:

{user_request}
"""
    
    # Format the prompt with the user's message
    formatted_prompt = prompt_template.format(user_request=latest_message)
    
    # Get response from model
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=config.get('temperature', 0.6) if config else 0.6,
        top_p=config.get('top_p', 0.95) if config else 0.95,
        max_tokens=config.get('max_tokens', 32768) if config else 32768,
    )
    
    response_text = response["choices"][0]["message"]["content"]

    # Extract content between &lt;refined_prompt&gt; tags
    start_tag = "<refined_prompt>"
    end_tag = "</refined_prompt>"
    
    start_idx = response_text.rfind(start_tag)
    if start_idx == -1:
        return response_text  # Return full response if no opening tag found
    
    # Get position after opening tag
    content_start = start_idx + len(start_tag)
    
    # Look for closing tag
    end_idx = response_text.find(end_tag, content_start)
    
    enriched = ""
    if end_idx == -1:
        # No closing tag - return everything after opening tag
        enriched = response_text[content_start:]
    else:
        # Return content between tags
        enriched = response_text[content_start:end_idx]
    
    print("Original request:", latest_message)
    print("Enriched response:", enriched)
    return enriched
