"""
Universal CrekAI Verification Cell - Logic-Based Validation
============================================================
This cell captures ALL variables and the API validates the LOGIC, not names!

Students can name variables anything they want - we check if the OUTPUT is correct.

PASTE THIS ONCE, WORKS FOREVER!
"""

import requests
import json

# ===== CONFIGURATION =====
USER_TOKEN = "paste_your_token_here"
PROJECT_ID = "learn-numpy"  # Update for each notebook
STEP = 1                     # Update for each notebook
API_BASE_URL = "https://your-domain.com/api"
# =========================

print("🔍 CrekAI Smart Verification\n")
print(f"📚 Project: {PROJECT_ID}")
print(f"📊 Step: {STEP}\n")

# ===== CAPTURE ALL VARIABLES =====
def capture_variable_info(var_name, var_value):
    """Capture metadata about ANY variable"""
    info = {
        "name": var_name,
        "type": None,
        "value": None,
        "shape": None,
        "min": None,
        "max": None
    }
    
    # NumPy arrays
    try:
        import numpy as np
        if isinstance(var_value, np.ndarray):
            info["type"] = "numpy.ndarray"
            info["shape"] = list(var_value.shape)
            info["min"] = float(var_value.min())
            info["max"] = float(var_value.max())
            return info
    except:
        pass
    
    # PyTorch tensors
    try:
        import torch
        if isinstance(var_value, torch.Tensor):
            info["type"] = "torch.Tensor"
            info["shape"] = list(var_value.shape)
            info["min"] = float(var_value.min().item())
            info["max"] = float(var_value.max().item())
            return info
    except:
        pass
    
    # Numbers
    if isinstance(var_value, (int, float)):
        info["type"] = "float" if isinstance(var_value, float) else "int"
        info["value"] = float(var_value)
        return info
    
    # Lists/tuples
    if isinstance(var_value, (list, tuple)):
        info["type"] = "list" if isinstance(var_value, list) else "tuple"
        info["value"] = str(var_value)[:100]  # First 100 chars
        return info
    
    return info

# Capture ALL user-created variables
print("📊 Capturing all your variables...")
variables = {}

for var_name, var_value in list(globals().items()):
    # Skip built-ins and imports
    if var_name.startswith('_'):
        continue
    if var_name in ['In', 'Out', 'get_ipython', 'exit', 'quit', 'open', 'help']:
        continue
    if var_name in ['requests', 'json', 'np', 'torch', 'pd', 'plt']:
        continue
    if callable(var_value) and not hasattr(var_value, 'shape'):
        continue
    
    # Capture relevant variables
    try:
        info = capture_variable_info(var_name, var_value)
        if info["type"]:  # Only include if we detected a type
            variables[var_name] = info
            print(f"   ✓ {var_name} ({info['type']})")
    except:
        pass

print(f"\n✓ Captured {len(variables)} variables\n")

# ===== SUBMIT FOR VALIDATION =====
print("🚀 Submitting for validation...\n")

payload = {
    "token": USER_TOKEN,
    "project_id": PROJECT_ID,
    "step": STEP,
    "code": "executed",
    "output": {
        "variables": variables
    }
}

try:
    response = requests.post(
        f"{API_BASE_URL}/track-execution",
        json=payload,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print("=" * 70)
        print("✅ SUCCESS! Your code is correct!")
        print("=" * 70)
        print(f"\n{data.get('message', 'Assignment completed')}")
        if data.get('next_step'):
            print(f"\n🚀 Step {data['next_step']} unlocked! Open the next notebook.")
        print("\n👉 Return to CrekAI to continue learning")
        print("=" * 70)
    else:
        print("=" * 70)
        print(f"❌ Validation Failed")
        print("=" * 70)
        try:
            error_data = response.json()
            print(f"\n{error_data.get('error', 'Unknown error')}")
            if 'message' in error_data:
                print(f"\n💡 {error_data['message']}")
            print("\n🔍 Check your code logic and try again")
        except:
            print("\nPlease check your code and try again.")
        print("=" * 70)
        
except requests.exceptions.RequestException as e:
    print(f"❌ Network error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
