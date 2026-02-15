## Training data folders explained

training_data:
Original training data
Not the correct LLM training data format

training_data_formatted:
Same content as the "training_data" folder but formatted for LLMs

training_data_mistral_format:
This training data is pre-tokenized and should probably not be used
Was used to train the second version of the mistral-nemo LoRA model (mistral-nemo-cve)


training_data_tool_calls:
Data for how to use the qdrant tool
currently only has user query -> RAG query data, missing RAG return data and the handling of the returned data


training_data_explain_cve:
This is for teaching correct behavior when a user asks for a CVE summary by its ID.