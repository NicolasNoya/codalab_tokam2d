#!/usr/bin/env python3
"""
Hugging Face Login Script

This script will help you authenticate with Hugging Face to access gated models like DINOv3.

Instructions:
1. Get your token from: https://huggingface.co/settings/tokens
2. Run this script: python huggingface_login.py
3. Paste your token when prompted
4. Request access to DINOv3 at: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m

Your token will be saved securely in ~/.cache/huggingface/token
"""

from huggingface_hub import login

print("=" * 70)
print("Hugging Face Authentication")
print("=" * 70)
print()
print("To use gated models like DINOv3, you need to authenticate.")
print()
print("Steps:")
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Create a new token (or copy an existing one)")
print("3. Paste it below when prompted")
print()
print("After login, request access to DINOv3:")
print("   https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m")
print()
print("=" * 70)
print()

try:
    # This will prompt for the token interactively
    login()
    print()
    print("✓ Successfully logged in to Hugging Face!")
    print()
    print("Next steps:")
    print("1. Request access to DINOv3 at:")
    print("   https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m")
    print("2. Wait for approval (usually instant)")
    print("3. Run your notebook or training script")
    print()
except Exception as e:
    print()
    print(f"✗ Error during login: {e}")
    print()
    print("Alternative: Use environment variable")
    print("  export HUGGING_FACE_HUB_TOKEN='your_token_here'")
    print()
