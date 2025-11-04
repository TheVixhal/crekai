import Image from "next/image"

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
  const formatDate = (dateString?: string) => {
    if (!dateString) return "Recently"
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
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
          width={140} 
          height={140}
          className="opacity-40"
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
          <p className="text-xs font-medium text-gray-500 mb-1">Name</p>
          <p className="text-sm text-gray-900">{userProfile?.full_name || "N/A"}</p>
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
