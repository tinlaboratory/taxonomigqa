from huggingface_hub import login
from huggingface_hub import upload_folder
from huggingface_hub import create_repo

# Replace "YOUR_TOKEN" with the token you generated on the Hugging Face website
login(token="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")
# Replace "your_username/my_dataset" with the desired name of your dataset
repo_name = "yuluqinn24/fgqa_qs"

# Create a private repository by setting private=True
create_repo(repo_name, repo_type="dataset", private=True)
# Add license to the metadata
metadata = {
    "license": "cc-by-4.0"  # Replace with your chosen license (e.g., "cc0-1.0", "odbl-1.0")
}

# Path to your local dataset folder
local_folder_path = "/projectnb/tin-lab/yuluq/data/rgqa"

# Upload the entire folder to the dataset repository
upload_folder(
    folder_path=local_folder_path,
    repo_id=repo_name,
    repo_type="dataset"
)
