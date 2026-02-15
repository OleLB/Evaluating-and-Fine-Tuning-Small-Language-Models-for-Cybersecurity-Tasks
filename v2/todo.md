# Run the tool call queries throuhg vector database and get answers from chatGPT
    I have 400 queries ready to go
# Create traning data for when to not do tool calls
    Need traning data samples where the users asks about a CVE, the CVE data is supplied and the model gives a summary
# Train model again
    Already updated traning script to handle mistral better (hopefully) (LoRA/LoRA_fine_tune_geminiFix.py) <- it should be possible to pass json data to this script
    The updated file should work with both normal calls and tool calls
    This is the data format the traning data should be in:

    

