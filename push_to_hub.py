from huggingface_hub import HfApi, HfFolder, Repository, create_repo
import os
import time

def push_to_hub(repo_name=None, token=None, commit_message="Upload model", 
                local_dir="./temp_model", private=False):
    """Push a folder to Hugging Face Hub"""
    try:
        if not repo_name:
            raise ValueError("must specify a repo name to push to hub")
        
        if not os.path.exists(local_dir):
            raise ValueError(f"local_dir {local_dir} does not exist")
        
        print(f"Pushing model to the hub at {repo_name}...")
        api = HfApi()
        create_repo(repo_name, token=token, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_name, 
            token=token, 
            commit_message=commit_message
        )

        print(f"Model has been pushed to the hub at: {repo_name}")
        return True
        
    except Exception as e:
        print(f"Error pushing to hub: {str(e)}")
        return False

def push_zip_to_hub(repo_name=None, token=None, zip_file=None, 
                    commit_message="Upload file", private=False, repo_type="model",
                    max_retries=3, timeout=None):
    """
    Push a zip file to Hugging Face Hub with retry logic for large files
    
    Args:
        repo_name: Repository ID (username/repo-name)
        token: Hugging Face API token
        zip_file: Path to zip file to upload
        commit_message: Commit message for upload
        private: Make repo private
        repo_type: "model" or "dataset"
        max_retries: Number of retry attempts for connection errors
        timeout: HTTP timeout in seconds (None = no timeout)
    """
    try:
        if not repo_name:
            raise ValueError("must specify a repo name to push to hub")
        
        if not zip_file or not os.path.exists(zip_file):
            raise ValueError(f"zip_file {zip_file} does not exist")
        
        file_size_gb = os.path.getsize(zip_file) / (1024**3)
        print(f"Pushing zip file to the hub at {repo_name}...")
        print(f"File size: {file_size_gb:.2f} GB")
        
        api = HfApi()
        
        # Create repo if it doesn't exist
        create_repo(repo_name, token=token, private=private, exist_ok=True, repo_type=repo_type)
        
        # Retry logic for large file uploads
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1}/{max_retries}...")
                
                # Upload with custom timeout for large files
                upload_timeout = timeout or (3600 if file_size_gb > 1 else 300)
                
                api.upload_file(
                    path_or_fileobj=zip_file,
                    path_in_repo=os.path.basename(zip_file),
                    repo_id=repo_name,
                    token=token,
                    commit_message=commit_message,
                    repo_type=repo_type
                )
                
                print(f"\n✓ Zip file has been pushed successfully to: {repo_name}")
                return True
                
            except (ConnectionError, TimeoutError, Exception) as e:
                error_msg = str(e)
                
                # Check if it's a connection/timeout error
                is_connection_error = any(keyword in error_msg.lower() for keyword in 
                                         ['connection', 'timeout', 'remote closed', 'disconnected'])
                
                if is_connection_error and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠ Connection error: {error_msg}")
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
    except Exception as e:
        print(f"\n✗ Error pushing to hub: {str(e)}")
        print("\nTroubleshooting tips:")
        print("  1. Check your internet connection")
        print("  2. Try splitting the zip file into smaller chunks")
        print("  3. Use: python split_zip.py <zip_file> <chunk_size_mb>")
        return False
    
if __name__ == "__main__":
    # push_to_hub(
    #     repo_name="dangnguyens1/meta_train",
    #     token="",
    #     local_dir="/home/dang.nh4/VLM_Embed/meta_train/"
    # )
    
    push_zip_to_hub(
        repo_name="dangnguyens1/teacher_gradients",
        token="",
        zip_file="/mnt/disk1/backup_user/dang.nh4/VLM_Embed/qwenvl_2b_cls_vqa_grad.zip",
        commit_message="Upload zip file"
    )