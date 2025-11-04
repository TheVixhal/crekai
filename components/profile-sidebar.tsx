interface ProfileSidebarProps {
  user: {
    email?: string
  }
  userProfile?: {
    full_name?: string
    username?: string
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
  projectsEnrolled,
  completedSteps,
  projectsInProgress,
}: ProfileSidebarProps) {
  const formatDate = (dateString?: string) => {
    if (!dateString) return "Recently"
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
  }

  return (
    <div className="bg-white/70 backdrop-blur-md border border-gray-200 rounded-2xl p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 mb-6 tracking-tight">
        Your Profile
      </h3>

      {/* User Info */}
      <div className="space-y-4 mb-8">
        <div>
          <p className="text-xs uppercase tracking-wider text-gray-500 font-medium">Name</p>
          <p className="text-sm text-gray-800">{userProfile?.full_name || "N/A"}</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-gray-500 font-medium">Username</p>
          <p className="text-sm text-gray-800">{userProfile?.username || "N/A"}</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-gray-500 font-medium">Email</p>
          <p className="text-sm text-gray-800 break-words">{user.email}</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-gray-500 font-medium">Member Since</p>
          <p className="text-sm text-gray-800">{formatDate(createdAt)}</p>
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-gray-200 mb-6" />

      {/* Stats */}
      <div>
        <p className="text-xs uppercase tracking-wider text-gray-500 font-medium mb-4">
          Your Stats
        </p>

        <div className="space-y-4">
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 border border-gray-200 rounded-xl p-4">
            <p className="text-xs text-gray-600 uppercase font-medium mb-1">
              Projects Enrolled
            </p>
            <p className="text-2xl font-semibold text-gray-900">{projectsEnrolled}</p>
          </div>

          <div className="bg-gradient-to-r from-gray-50 to-gray-100 border border-gray-200 rounded-xl p-4">
            <p className="text-xs text-gray-600 uppercase font-medium mb-1">
              Steps Completed
            </p>
            <p className="text-2xl font-semibold text-gray-900">{completedSteps}</p>
          </div>

          <div className="bg-gradient-to-r from-gray-50 to-gray-100 border border-gray-200 rounded-xl p-4">
            <p className="text-xs text-gray-600 uppercase font-medium mb-1">
              In Progress
            </p>
            <p className="text-2xl font-semibold text-gray-900">{projectsInProgress}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
