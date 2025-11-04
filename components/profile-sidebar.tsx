"use client"
import Image from "next/image"
import { useState } from "react"

interface ProfileSidebarProps {
  user: {
    email?: string
  }
  userProfile?: {
    full_name?: string
    username?: string
    level?: number
  }
  createdAt?: string
  projectsEnrolled: number
  completedSteps: number
  projectsInProgress: number
}

export default function ProfileSidebar({
  user,
  userProfile,
  createdAt,
}: ProfileSidebarProps) {
  const [isEditingName, setIsEditingName] = useState(false)
  const [name, setName] = useState(userProfile?.full_name || "")
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState("")

  const formatDate = (dateString?: string) => {
    if (!dateString) return "Recently"
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
  }

  const handleSaveName = async () => {
    if (!name.trim()) {
      setError("Name cannot be empty")
      return
    }

    setSaving(true)
    setError("")

    try {
      const response = await fetch('/api/profile/update-name', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fullName: name }),
      })

      if (!response.ok) {
        throw new Error('Failed to update name')
      }

      setIsEditingName(false)
      window.location.reload()
    } catch (err) {
      setError("Failed to update name")
      console.error(err)
    } finally {
      setSaving(false)
    }
  }

  const userLevel = userProfile?.level ?? 0
  const subscriptionStatus = userLevel === 1 ? "Pro" : "Free"

  return (
    <div className="space-y-6">
      {/* Chameleon Background Header */}
      <div className="relative h-32 bg-gray-50 rounded-xl flex items-center justify-center overflow-hidden">
        <Image 
          src="/chameleon.png" 
          alt="Chameleon" 
          width={280} 
          height={140}
          className="opacity-100"
        />
      </div>

      {/* Subscription Status */}
      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-gray-500 mb-1">Subscription</p>
            <p className="text-lg font-semibold text-gray-900">{subscriptionStatus}</p>
          </div>
          {userLevel === 0 && (
            <a 
              href="/subscription" 
              className="px-4 py-2 bg-black text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors"
            >
              Upgrade
            </a>
          )}
          {userLevel === 1 && (
            <div className="flex items-center justify-center w-10 h-10 bg-black rounded-lg">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
              </svg>
            </div>
          )}
        </div>
      </div>

      {/* User Info */}
      <div className="space-y-4">
        <div>
          <div className="flex items-center justify-between mb-1">
            <p className="text-xs font-medium text-gray-500">Name</p>
            {!isEditingName && (
              <button 
                onClick={() => setIsEditingName(true)}
                className="text-xs text-blue-600 hover:text-blue-800 font-medium"
              >
                Edit
              </button>
            )}
          </div>
          
          {isEditingName ? (
            <div className="space-y-2">
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
                placeholder="Enter your name"
              />
              {error && <p className="text-xs text-red-600">{error}</p>}
              <div className="flex gap-2">
                <button
                  onClick={handleSaveName}
                  disabled={saving}
                  className="px-3 py-1.5 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors disabled:opacity-50"
                >
                  {saving ? "Saving..." : "Save"}
                </button>
                <button
                  onClick={() => {
                    setIsEditingName(false)
                    setName(userProfile?.full_name || "")
                    setError("")
                  }}
                  className="px-3 py-1.5 bg-gray-100 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-900">{userProfile?.full_name || "N/A"}</p>
          )}
        </div>
        
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1">Username</p>
          <p className="text-sm text-gray-900">{userProfile?.username || "N/A"}</p>
        </div>
        
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1">Email</p>
          <p className="text-sm text-gray-900 break-words">{user.email}</p>
        </div>
        
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1">Member Since</p>
          <p className="text-sm text-gray-900">{formatDate(createdAt)}</p>
        </div>
      </div>
    </div>
  )
}
