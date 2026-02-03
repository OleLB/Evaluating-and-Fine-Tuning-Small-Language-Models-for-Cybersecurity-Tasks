
# Data prep
1. Distillation data (use 1000 CVE descriptions from database)
    Figure out what ChatGPT is supposed to do
    Create a prompt
    Make a script that:
        Finds 1000 random CVEs from local database (select random int in database entry range)
        for each selected CVE:
            get its description
            send the query
            save response to database?

2. Get 9000 records from alpaca dataset

# LoRA
1. Write a script to do LoRA traning with the dataset (on Mistral-Nemo)

# Experimental
Dont show qdrant results to user and show top 20 to the AI? (Mistral-Nemo has a large context window)
