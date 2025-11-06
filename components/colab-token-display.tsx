"use client"

import { useEffect, useState } from "react"
import { Copy, RefreshCw, Eye, EyeOff, ChevronDown, ChevronUp } from "lucide-react"

export default function ColabTokenDisplay({ projectSlug }: { projectSlug: string }) {
  const [token, setToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [copied, setCopied] = useState(false)
  const [showToken, setShowToken] = useState(false)
  const [error, setError] = useState("")
  const [isExpanded, setIsExpanded] = useState(false)

  useEffect(() => {
    fetchToken()
  }, [])

  // ========== FETCH TOKEN ==========
  const fetchToken = async () => {
    setLoading(true)
    setError("")
    try {
      const response = await fetch("/api/profile/get-token")
      const data = await response.json()

      if (data.success) {
        setToken(data.token)
      } else {
        setError("Failed to load token")
      }
    } catch (err) {
      console.error(err)
      setError("Failed to load token")
    } finally {
      setLoading(false)
    }
  }

  // ========== GENERATE NEW TOKEN ==========
  const generateToken = async () => {
    setGenerating(true)
    setError("")
    try {
      const response = await fetch("/api/profile/generate-token", {
        method: "POST",
      })
      const data = await response.json()

      if (data.success) {
        setToken(data.token)
      } else {
        setError("Failed to generate token")
      }
    } catch (err) {
      console.error(err)
      setError("Failed to generate token")
    } finally {
      setGenerating(false)
    }
  }

  // ========== UNIFIED CODE GENERATOR ==========
  const generateTrackingCode = (token: string, projectSlug: string) => `"""
Universal CrekAI Verification Cell
Copy this into your Colab notebook - it auto-captures all variables!
"""

import requests
import json

# ===== CONFIGURATION =====
USER_TOKEN = "${token}"
PROJECT_ID = "${projectSlug}"
STEP = 1  # Update this for each step
API_BASE_URL = "https://your-domain.com/api"
# =========================

print("🔍 CrekAI Verification\\n")

# ===== AUTO-CAPTURE ALL VARIABLES =====
def capture_variable_info(var_name, var_value):
    info = {"name": var_name, "type": None, "value": None, "shape": None, "min": None, "max": None}
    
    # NumPy arrays
    try:
        import numpy as np
        if isinstance(var_value, np.ndarray):
            info["type"] = "numpy.ndarray"
            info["shape"] = list(var_value.shape)
            info["min"] = float(var_value.min())
            info["max"] = float(var_value.max())
            return info
    except: pass
    
    # PyTorch tensors
    try:
        import torch
        if isinstance(var_value, torch.Tensor):
            info["type"] = "torch.Tensor"
            info["shape"] = list(var_value.shape)
            info["min"] = float(var_value.min().item())
            info["max"] = float(var_value.max().item())
            return info
    except: pass
    
    # Numbers
    if isinstance(var_value, (int, float)):
        info["type"] = "float" if isinstance(var_value, float) else "int"
        info["value"] = float(var_value)
        return info
    
    return info

# Capture ALL variables
print("📊 Capturing variables...")
variables = {}

for var_name, var_value in list(globals().items()):
    if var_name.startswith('_'): continue
    if var_name in ['In', 'Out', 'get_ipython', 'exit', 'quit', 'requests', 'json', 'np', 'torch']: continue
    if callable(var_value) and not hasattr(var_value, 'shape'): continue
    
    try:
        info = capture_variable_info(var_name, var_value)
        if info["type"]:
            variables[var_name] = info
            print(f"   ✓ {var_name}")
    except: pass

# ===== SUBMIT =====
print("\\n🚀 Submitting...\\n")

try:
    response = requests.post(
        f"{API_BASE_URL}/track-execution",
        json={
            "token": USER_TOKEN,
            "project_id": PROJECT_ID,
            "step": STEP,
            "code": "executed",
            "output": {"variables": variables}
        },
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print("=" * 60)
        print("✅ SUCCESS! Assignment Verified!")
        print("=" * 60)
        print(f"\\n{data.get('message', '')}")
        if data.get('already_completed'):
            print("\\n✅ Step already completed — re-verified successfully!")
        if data.get('next_step'):
            print(f"\\n🚀 Step {data['next_step']} unlocked!")
        print("\\n👉 Return to CrekAI")
        print("=" * 60)
    else:
        print("❌ Validation Failed")
        error_data = response.json()
        print(f"\\n{error_data.get('message', 'Check your code')}")

except Exception as e:
    print(f"❌ Error: {e}")
`

  // ========== COPY CODE ==========
  const copyTrackingCode = () => {
    if (!token) return
    const code = generateTrackingCode(token, projectSlug)
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const copyToken = () => {
    if (token) {
      navigator.clipboard.writeText(token)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  // ========== RENDER ==========
  if (loading) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <div className="text-blue-700 text-sm">Loading token...</div>
      </div>
    )
  }

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg mb-6 overflow-hidden">
      {/* HEADER */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-blue-100 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="text-2xl">🔗</div>
          <div className="text-left">
            <h3 className="text-base font-semibold text-blue-900">
              Colab Assignment Verification
            </h3>
            <p className="text-xs text-blue-700">
              {token ? "Token ready - Click to view" : "Generate token to track assignments"}
            </p>
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-blue-700" />
        ) : (
          <ChevronDown className="w-5 h-5 text-blue-700" />
        )}
      </button>

      {/* BODY */}
      {isExpanded && (
        <div className="px-6 pb-6 pt-2 border-t border-blue-200">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-red-700 text-sm">
              {error}
            </div>
          )}

          {!token ? (
            <div className="space-y-3">
              <p className="text-sm text-blue-800">
                Generate your unique token to start tracking your Colab assignments.
              </p>
              <button
                onClick={generateToken}
                disabled={generating}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition text-sm font-medium flex items-center gap-2"
              >
                <RefreshCw className={`w-4 h-4 ${generating ? "animate-spin" : ""}`} />
                {generating ? "Generating..." : "Generate Token"}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {/* TOKEN DISPLAY */}
              <div>
                <label className="text-sm font-medium text-blue-900 mb-2 block">
                  Your Token:
                </label>
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-white border border-blue-300 rounded-lg p-3 font-mono text-sm break-all">
                    {showToken ? token : "•".repeat(32)}
                  </div>
                  <button
                    onClick={() => setShowToken(!showToken)}
                    className="p-2 bg-white border border-blue-300 rounded-lg hover:bg-blue-50 transition"
                    title={showToken ? "Hide token" : "Show token"}
                  >
                    {showToken ? (
                      <EyeOff className="w-5 h-5 text-blue-700" />
                    ) : (
                      <Eye className="w-5 h-5 text-blue-700" />
                    )}
                  </button>
                  <button
                    onClick={copyToken}
                    className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                    title="Copy token"
                  >
                    <Copy className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* TRACKING CODE DISPLAY */}
              <div className="border-t border-blue-200 pt-4">
                <p className="text-sm text-blue-800 mb-3 font-medium">
                  📋 Universal Verification Code (Auto-captures all variables):
                </p>
                <div className="bg-gray-900 rounded-lg p-4 text-sm font-mono text-green-400 relative overflow-x-auto max-h-96 overflow-y-auto">
                  <pre className="whitespace-pre-wrap break-all text-xs">
                    {generateTrackingCode(token, projectSlug)}
                  </pre>
                </div>
                <button
                  onClick={copyTrackingCode}
                  className="mt-3 px-4 py-2 bg-white border border-blue-300 text-blue-700 rounded-lg hover:bg-blue-50 transition text-sm font-medium flex items-center gap-2"
                >
                  <Copy className="w-4 h-4" />
                  {copied ? "Copied!" : "Copy Full Code"}
                </button>
                <p className="text-xs text-blue-600 mt-2">
                  ℹ️ Code auto-captures ALL variables - no manual list needed! Just update STEP number.
                </p>
              </div>

              {/* REGENERATE TOKEN */}
              <div className="flex items-center gap-2">
                <button
                  onClick={generateToken}
                  disabled={generating}
                  className="px-3 py-1.5 bg-white border border-blue-300 text-blue-700 rounded-lg hover:bg-blue-50 disabled:opacity-50 transition text-sm flex items-center gap-2"
                >
                  <RefreshCw className={`w-3 h-3 ${generating ? "animate-spin" : ""}`} />
                  {generating ? "Regenerating..." : "Regenerate Token"}
                </button>
                <span className="text-xs text-blue-600">
                  Regenerating will invalidate your old token
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
