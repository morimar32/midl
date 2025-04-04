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
    if not enriched_content.strip():
        print("No enriched content generated. Returning original message.")
        return "No enriched content generated. Returning original message."
    
    # Generate expert persona based on enriched content
    expert_persona = generate_expert(enriched_content, config)
    if not expert_persona.strip():
        print("No expert persona generated. Returning enriched content.")
        return "No expert persona generated. Returning enriched content."
    
    final_prompt = build_final_prompt(enriched_content, expert_persona)
    formatted_messages.append({"role": final_prompt, "content": enrich_request})
    #formatted_messages.append({"role": messages[-1].role, "content": final_prompt})
    
    # Get response from model
    response = llm.create_chat_completion(
        messages=formatted_messages,
        temperature=config.get('temperature', 0.6) if config else 0.6,
        top_p=config.get('top_p', 0.95) if config else 0.95,
        max_tokens=config.get('max_tokens', 32768) if config else 32768,
    )
    
    # Combine expert persona with model response
    model_response = response["choices"][0]["message"]["content"]
    return model_response

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

def generate_expert(enriched_context: str, config: dict = None) -> str:
    """
    Generate an expert persona based on enriched context by calling the LLM.
    
    Args:
        enriched_context: The enriched context to generate expert persona from
        config: Optional configuration for the LLM call
        
    Returns:
        The ideal expert persona extracted from the LLM response
    """
    if llm is None:
        raise RuntimeError("Model not initialized. Call initialize() first.")
    
    # Prompt template to generate expert persona
    prompt_template = """<prompt>
  <task>Analyze the following sample prompt thoughtfully and reflect on its characteristics to determine the best approach for answering it. **Think step by step and do a thorough job!**</task>
  <instruction>
    You are an expert in prompt analysis. Your goal is to understand the structure and intent of a given prompt so that you can identify its domain, the ideal expert to answer it, and the specific information needed for a comprehensive response. **Your detailed and accurate analysis is highly valued.**
    MAKE SURE YOU RETURN THE INFORMATION IN XML FORMAT, specifically within the `<reflection_points>` tags.
  </instruction>
  <positive_encouragement>
    We believe in your ability to provide a comprehensive and well-structured analysis. **Take your time and ensure accuracy in each step of your reasoning.**
  </positive_encouragement>
  <reward_system>
    **You will be recognized for:**
    <reward>Providing specific and well-justified answers for the domain and ideal expert.</reward>
    <reward>Listing comprehensive and relevant types of needed information.</reward>
    <reward>Ensuring your response is correctly formatted within the `<reflection_points>` XML structure.</reward>
    **Accuracy and attention to detail will be highly rewarded in our assessment.**
  </reward_system>
  <chain_of_thought>
    <step>First, carefully read and understand the content of the <sample_prompt> provided below.</step>
    <step>Next, based on the keywords, concepts, and the nature of the question, determine the **primary domain** or subject area to which the sample prompt belongs. Consider fields like science, history, technology, literature, etc.</step>
    <step>Then, think about the type of expertise required to provide a knowledgeable and accurate answer to the sample prompt. Identify the **specific type of expert** (e.g., a software engineer specializing in Python, a historian focused on the Roman Empire, a biologist studying genetics). Make sure the type of expert is well qualified, and could teach an advanced course in the subject matter. Make sure that you are only selecting ONE type of expert. If there multiple options, pick the best one. **Justify your choice with a clear and detailed explanation based directly on the content of the prompt.**</step>
    <step>Finally, consider the kind of information that would be most relevant and necessary to construct a thorough and effective answer to the sample prompt. List the **specific types of information** that an expert would need to draw upon (e.g., definitions of key terms, historical dates, technical specifications, relevant theories, step-by-step procedures). **Be specific and consider the level of detail an expert in the field would need.**</step>
    <step>Present your analysis by filling in the sections accurately and completely under <reflection_points>.</step>
  </chain_of_thought>
  <sample_prompt>
    [SAMPLE PROMPT WILL BE INSERTED HERE BY THE USER]
  </sample_prompt>
  <hallucination_warning>
    **It is critically important that you do not hallucinate or invent information.** Only provide analysis that is directly supported by your understanding of the sample prompt and general knowledge. If you are uncertain about any aspect, it is better to acknowledge the uncertainty than to provide potentially incorrect or fabricated details. **Accuracy is paramount.**
  </hallucination_warning>
  <reflection_points>
    <domain>[To be filled by the LLM]</domain>
    <ideal_expert>[To be filled by the LLM] (Justification: [To be filled by the LLM])</ideal_expert>
    <needed_information>
      <item>[To be filled by the LLM]</item>
      <item>[To be filled by the LLM]</item>
      <item>[To be filled by the LLM]</item>
      <item>[To be filled by the LLM]</item>
    </needed_information>
  </reflection_points>
</prompt>

BELOW IS THE SAMPLE PROMPT: 

{context}
"""
    
    # Format the prompt with the enriched context
    formatted_prompt = prompt_template.format(context=enriched_context)
    
    # Get response from model
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=config.get('temperature', 0.6) if config else 0.6,
        top_p=config.get('top_p', 0.95) if config else 0.95,
        max_tokens=config.get('max_tokens', 32768) if config else 32768,
    )
    
    response_text = response["choices"][0]["message"]["content"]
    
    # First extract reflection_points block
    reflection_start = response_text.rfind("<reflection_points>")
    if reflection_start == -1:
        print("Return empty if no reflection points found")
        print("***")
        print(response_text)
        return ""  # Return empty if no reflection points found
    
    reflection_end = response_text.rfind("</reflection_points>", reflection_start)
    if reflection_end == -1:
        reflection_block = response_text[reflection_start:]
    else:
        reflection_block = response_text[reflection_start:reflection_end]
    
    # Then extract first ideal_expert from reflection block
    expert_start = reflection_block.find("<ideal_expert>")
    if expert_start == -1:
        print("Return empty if no expert tag found")
        print("***")
        print(response_text)
        return ""  # Return empty if no expert tag found
    
    expert_end = reflection_block.find("</ideal_expert>", expert_start)
    if expert_end == -1:
        expert = reflection_block[expert_start + len("<ideal_expert>"):]
        print(expert)
        return expert
    
    expert = reflection_block[expert_start + len("<ideal_expert>"):expert_end]
    print("expert:", expert)
    return expert

def build_final_prompt(enriched_content: str, expert_persona: str) -> str:
    """
    Combine enriched content and expert persona into a final prompt for the LLM.
    
    Args:
        enriched_content: The enriched content from enrich_request()
        expert_persona: The expert persona from generate_expert()
        
    Returns:
        The final formatted prompt combining all components
    """
    return f"""<prompt>
  <context>
    <description>You are tasked with providing a detailed explanation to the user's inquiry. Please focus on the core mechanisms and key aspects.</description>
    <llm_guidance>As a smaller language model, we understand that your resources are limited. Therefore, we aim to provide a clear and concise prompt to help you deliver the best possible response.</llm_guidance>
  </context>
  <objective>
    <goal>To generate a comprehensive and accurate explanation that demonstrates a strong understanding of its fundamental principles.</goal>
  </objective>
  <specificity>
    <format_preference>Please structure your explanation in clear paragraphs, using bullet points if helpful to list key features or steps.</format_preference>
  </specificity>
  <persona>
    {expert_persona}
  </persona>
  <task>
    <action>Explain in detail the following:</action>
    <topic>{enriched_content}</topic>
  </task>
  <resources_constraints>
    <efficiency_focus>Please be as concise as possible while ensuring sufficient detail.</efficiency_focus>
    <accuracy_emphasis>
      <warning>
        **It is absolutely critical that the information you provide is accurate and based on verifiable knowledge. Please do not generate or include any information that is fabricated, speculative, or constitutes a hallucination.** 
      </warning>
      <instruction>If you are uncertain about any aspect of the topic, please explicitly state your uncertainty rather than providing potentially incorrect information. </instruction>
    </accuracy_emphasis>
    <output_format_constraint>Your response should be formatted in well-structured paragraphs, potentially using bullet points for lists. Tables are also an option, if relevant and useful to the topic.</output_format_constraint>
  </resources_constraints>
  <reward_system>
    <positive_reinforcement>A highly detailed, accurate, and well-explained response that adheres to these instructions and avoids any hallucinations will be considered a top-quality result. Such a response will be greatly appreciated and acknowledged as an excellent demonstration of your capabilities.</positive_reinforcement> 
    <quality_criteria>Key indicators of a high-quality response include:</quality_criteria>
    <ul>
      <li>Demonstrated in-depth understanding of the topic.</li>
      <li>Accuracy of all provided information.</li>
      <li>Clarity and logical flow of the explanation.</li>
      <li>Appropriate level of detail without being overly verbose.</li>
      <li>Complete absence of hallucinations or fabricated information.</li>
    </ul>
  </reward_system>
</prompt>"""
