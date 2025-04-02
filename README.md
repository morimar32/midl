# midl

## Introduction

midl is a OpenAI compatible REST endpoint, which is meant to act as a middle-man for other LLM calls. Local LLMs are loaded through llama.cpp, with planned support for draft models for speculative decoding. Initial planned external LLM support is what you would expect (OpenAI, Anthropic, Deepseek, Gemini). 

## Interception Tools

### Round 1 Tools

These are simpler, more general  tools that will serve as building blocks and discovery for more complex and bespoke tooling

  - enrichr - experiment for small LLM to improve the quality of the results by chaining multiple internal LLM calls with specially crafted prompts before actually calling the LLM
	  - The original request "enhanced" with a purpose built prompt and local LLM call to infer more specifics 
	  - The enhanced request is passed to a second purpose built prompt meant to extract out the domain, and ideal expert who could best respond
	  - That additional information is used to make a specialized persona, that then responds back to the enhanced request, and that response is sent back to the user
  - recordr - logs each request and subsequent response to a local Sqlite table. It will record a hash of each individual message, and will record it only once. It will use the sequence of identical message hashes to establish a sequence. Its table structure is meant for subsequent analysis for context optimization, as well as supporting the creation of future, more complex interceptors as well as message classification
  - summarizr - context window optimizer meant to summarize older parts of chat history using a small local LLM
  - rag - good ol' retrieval augmentation generation. probably will have a companion tool for loading documents

### Round 2 Tools

This list is more theoretical and will likely change and grow as more experiments are done

  - roo context flush - specific to ai based coding, where the chat history is trimmed of unneeded intermediary steps such as old runs of tools and unit tests, and other iterative steps where a minor issue was found and resolved
  - roo prefiller - analyze certain prompts from roo code and act like a project specific RAG, automatically pre-filling the context with additional information
  - enrichr v2 - much more robust version of the original enrichr for small LLMs.


## Random Notes
* `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --break-system-packages`

