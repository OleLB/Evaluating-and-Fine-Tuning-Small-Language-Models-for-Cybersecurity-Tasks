#!/usr/bin/env python3
"""
Qdrant Docker Container Manager
Cross-platform script for managing Qdrant vector database container
"""

import subprocess
import sys
import platform
import time

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    print("⚠ qdrant-client library not found")
    print("  Please install it: pip install qdrant-client")
    sys.exit(1)

# Configuration
QDRANT_IMAGE = "qdrant/qdrant:latest"
CONTAINER_NAME = "security_assistant_qdrant"
QDRANT_PORT = "6333"
GRPC_PORT = "6334"
QDRANT_URL = f"http://localhost:{QDRANT_PORT}"


def check_qdrant_available():
    """Check if Qdrant is already running and accessible."""
    print("Checking if Qdrant is available...")
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        # Try to get collections to verify connection
        client.get_collections()
        print(f"✓ Qdrant is already running at {QDRANT_URL}")
        return True
    except Exception as e:
        print(f"✗ Qdrant is not available: {type(e).__name__}")
        return False


def run_command(cmd, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            check=False
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def check_docker_installed():
    """Check if Docker is installed and accessible."""
    print("\nChecking if Docker is installed...")
    returncode, stdout, stderr = run_command("docker --version")
    
    if returncode == 0:
        print(f"✓ Docker is installed: {stdout}")
        return True
    else:
        print(f"✗ Docker is not installed or not accessible")
        print(f"  Error: {stderr}")
        return False


def check_image_exists():
    """Check if the Qdrant image exists locally."""
    print(f"\nChecking if image '{QDRANT_IMAGE}' exists...")
    returncode, stdout, _ = run_command(f"docker images -q {QDRANT_IMAGE}")
    
    if stdout:
        print(f"✓ Image '{QDRANT_IMAGE}' exists")
        return True
    else:
        print(f"✗ Image '{QDRANT_IMAGE}' does not exist")
        return False


def pull_image():
    """Pull the Qdrant Docker image."""
    print(f"\nPulling image '{QDRANT_IMAGE}'...")
    returncode, stdout, stderr = run_command(f"docker pull {QDRANT_IMAGE}", capture_output=False)
    
    if returncode == 0:
        print(f"✓ Successfully pulled image '{QDRANT_IMAGE}'")
        return True
    else:
        print(f"✗ Failed to pull image '{QDRANT_IMAGE}'")
        print(f"  Error: {stderr}")
        return False


def check_container_exists():
    """Check if the Qdrant container exists."""
    print(f"\nChecking if container '{CONTAINER_NAME}' exists...")
    returncode, stdout, _ = run_command(f"docker ps -a -q -f name=^{CONTAINER_NAME}$")
    
    if stdout:
        print(f"✓ Container '{CONTAINER_NAME}' exists")
        return True
    else:
        print(f"✗ Container '{CONTAINER_NAME}' does not exist")
        return False


def is_container_running():
    """Check if the container is currently running."""
    print(f"\nChecking if container '{CONTAINER_NAME}' is running...")
    returncode, stdout, _ = run_command(f"docker ps -q -f name=^{CONTAINER_NAME}$")
    
    if stdout:
        print(f"✓ Container '{CONTAINER_NAME}' is running")
        return True
    else:
        print(f"✗ Container '{CONTAINER_NAME}' is not running")
        return False


def create_and_start_container():
    """Create and start a new Qdrant container."""
    print(f"\nCreating and starting container '{CONTAINER_NAME}'...")
    
    cmd = (
        f"docker run -d "
        f"--name {CONTAINER_NAME} "
        f"-p {QDRANT_PORT}:{QDRANT_PORT} "
        f"-p {GRPC_PORT}:{GRPC_PORT} "
        f"-v qdrant_storage:/qdrant/storage "
        f"{QDRANT_IMAGE}"
    )
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print(f"✓ Successfully created and started container '{CONTAINER_NAME}'")
        print(f"  Container ID: {stdout}")
        return True
    else:
        print(f"✗ Failed to create container '{CONTAINER_NAME}'")
        print(f"  Error: {stderr}")
        return False


def start_container():
    """Start an existing container."""
    print(f"\nStarting container '{CONTAINER_NAME}'...")
    returncode, stdout, stderr = run_command(f"docker start {CONTAINER_NAME}")
    
    if returncode == 0:
        print(f"✓ Successfully started container '{CONTAINER_NAME}'")
        return True
    else:
        print(f"✗ Failed to start container '{CONTAINER_NAME}'")
        print(f"  Error: {stderr}")
        return False


def wait_for_qdrant(max_attempts=30, delay=1):
    """Wait for Qdrant to become available."""
    print("\nWaiting for Qdrant to be ready...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            client = QdrantClient(url=QDRANT_URL, timeout=2)
            client.get_collections()
            print(f"✓ Qdrant is ready!")
            return True
        except Exception:
            if attempt < max_attempts:
                print(f"  Attempt {attempt}/{max_attempts}... waiting {delay}s")
                time.sleep(delay)
            else:
                print(f"✗ Qdrant did not become available after {max_attempts} attempts")
                return False
    
    return False


def main():
    """Main execution flow."""
    print("=" * 60)
    print("Qdrant Docker Container Manager")
    print(f"Platform: {platform.system()}")
    print("=" * 60)
    
    # Check if Qdrant is already available
    if check_qdrant_available():
        print("\n" + "=" * 60)
        print("✓ Qdrant is ready! Nothing to do.")
        print(f"  REST API: {QDRANT_URL}")
        print(f"  gRPC API: http://localhost:{GRPC_PORT}")
        print(f"  Dashboard: {QDRANT_URL}/dashboard")
        print("=" * 60)
        return
    
    # If not available, proceed with Docker setup
    print("\nQdrant is not available, proceeding with Docker setup...")
    
    # Step 1: Check Docker installation
    if not check_docker_installed():
        print("\n⚠ Please install Docker and try again")
        sys.exit(1)
    
    # Step 2: Check if image exists
    if not check_image_exists():
        # Step 3: Pull image if it doesn't exist
        if not pull_image():
            print("\n⚠ Failed to pull image")
            sys.exit(1)
    
    # Step 4: Check if container exists
    if not check_container_exists():
        # Step 5: Create and start container if it doesn't exist
        if not create_and_start_container():
            print("\n⚠ Failed to create container")
            sys.exit(1)
    else:
        # Step 6: Check if container is running
        if not is_container_running():
            # Step 7: Start container if not running
            if not start_container():
                print("\n⚠ Failed to start container")
                sys.exit(1)
    
    # Wait for Qdrant to be ready
    if not wait_for_qdrant():
        print("\n⚠ Container started but Qdrant is not responding")
        print("  You may need to check the container logs:")
        print(f"  docker logs {CONTAINER_NAME}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Qdrant is ready!")
    print(f"  REST API: {QDRANT_URL}")
    print(f"  gRPC API: http://localhost:{GRPC_PORT}")
    print(f"  Dashboard: {QDRANT_URL}/dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()