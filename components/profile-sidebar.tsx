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
    <div className="bg-white border-2 border-black p-6 sticky top-24 h-fit">
      <h3 className="text-lg font-bold font-serif text-black mb-4 border-b-2 border-black pb-2">Your Profile</h3>

      {/* User Info */}
      <div className="mb-6">
        <div className="mb-3">
          <p className="text-xs uppercase tracking-widest text-gray-600 font-sans font-bold">Name</p>
          <p className="text-sm text-black font-sans">{userProfile?.full_name || "N/A"}</p>
        </div>
        <div className="mb-3">
          <p className="text-xs uppercase tracking-widest text-gray-600 font-sans font-bold">Username</p>
          <p className="text-sm text-black font-sans">{userProfile?.username || "N/A"}</p>
        </div>
        <div className="mb-3">
          <p className="text-xs uppercase tracking-widest text-gray-600 font-sans font-bold">Email</p>
          <p className="text-sm text-black font-sans break-words">{user.email}</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-widest text-gray-600 font-sans font-bold">Member Since</p>
          <p className="text-sm text-black font-sans">{formatDate(createdAt)}</p>
        </div>
      </div>

      <div className="border-t-2 border-black my-4"></div>

      {/* Stats */}
      <div>
        <p className="text-xs uppercase tracking-widest text-gray-600 font-sans font-bold mb-4">Your Stats</p>

        <div className="space-y-4">
          <div className="bg-amber-50 border-2 border-black p-3">
            <p className="text-xs uppercase tracking-widest text-gray-700 font-sans font-bold">Projects Enrolled</p>
            <p className="text-2xl font-bold text-black font-serif">{projectsEnrolled}</p>
          </div>

          <div className="bg-amber-50 border-2 border-black p-3">
            <p className="text-xs uppercase tracking-widest text-gray-700 font-sans font-bold">Steps Completed</p>
            <p className="text-2xl font-bold text-black font-serif">{completedSteps}</p>
          </div>

          <div className="bg-amber-50 border-2 border-black p-3">
            <p className="text-xs uppercase tracking-widest text-gray-700 font-sans font-bold">In Progress</p>
            <p className="text-2xl font-bold text-black font-serif">{projectsInProgress}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
