import fs from "fs"
import path from "path"
import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import ProjectCard from "@/components/project-card"
import ProfileSidebar from "@/components/profile-sidebar"

export default async function ProjectsPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth/login")
  }

  // 🔹 Fetch local markdown projects metadata
  const projectsDir = path.join(process.cwd(), "public", "projects")
  const projectFolders = fs.readdirSync(projectsDir)

  const projects = projectFolders.map((folder) => {
    const metaPath = path.join(projectsDir, folder, "meta.json")
    const meta = fs.existsSync(metaPath)
      ? JSON.parse(fs.readFileSync(metaPath, "utf-8"))
      : {
          slug: folder,
          title: folder,
          description: "No description available.",
          total_steps: 1,
        }
    return meta
  })

  // 🔹 Fetch user progress
  const { data: progressData } = await supabase
    .from("user_progress")
    .select("*")
    .eq("user_id", user.id)

  const progressMap = (progressData || []).reduce((acc: Record<string, any>, prog: any) => {
    acc[prog.project_id] = prog
    return acc
  }, {})

  const { data: userProfile } = await supabase
    .from("user_profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle()

  const projectsEnrolled = Object.keys(progressMap).length
  const completedSteps = Object.values(progressMap).reduce(
    (sum, prog: any) => sum + (prog.completed_steps?.length || 0),
    0
  )
  const projectsInProgress = Object.values(progressMap).filter(
    (prog: any) => prog.current_step && prog.current_step <= (prog.total_steps || 0)
  ).length

  // 🎨 Anthropic cream base + accents
  const cream = "#eeece2" // Background
  const mint = "#BED2CD" // Muted accent green
  const lavender = "#C8C6DA" // Soft neutral lavender-gray
  const blue = "#6594C1" // Calm blue accent

  return (
    <div
      className="min-h-screen text-gray-900"
      style={{ backgroundColor: cream }}
    >
      {/* Header */}
      <header
        className="sticky top-0 z-50 backdrop-blur-md border-b border-gray-300"
        style={{
          backgroundColor: `${lavender}D9`, // soft translucent lavender
        }}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight text-gray-900">CrekAI</h1>
            <span className="text-sm text-gray-700">Learning Projects</span>
          </div>
          <form
            action={async () => {
              "use server"
              const supabase = await createClient()
              await supabase.auth.signOut()
              redirect("/")
            }}
          >
            <button
              className="px-4 py-2 rounded-md text-sm font-medium text-white transition"
              style={{
                backgroundColor: blue,
              }}
            >
              Sign Out
            </button>
          </form>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-6 py-12 grid grid-cols-1 lg:grid-cols-3 gap-10">
        {/* Left Section */}
        <section className="lg:col-span-2 space-y-10">
          {/* Welcome Banner */}
          <div
            className="rounded-2xl border border-gray-200 shadow-sm p-8"
            style={{
              backgroundColor: mint,
              color: "#1F2A2C",
            }}
          >
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-semibold mb-2">
                  Welcome back, {userProfile?.full_name?.split(" ")[0] || "Learner"} 👋
                </h2>
                <p className="text-gray-800 text-sm">
                  Continue your AI/ML learning journey — progress grows one project at a time.
                </p>
              </div>
              <div className="hidden md:block text-5xl opacity-80">🚀</div>
            </div>
          </div>

          {/* Section Header */}
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-gray-900">Featured Projects</h3>
            <p className="text-gray-600 text-sm">
              Explore hands-on AI projects curated for practical learning.
            </p>
          </div>

          {/* Projects Grid */}
          <div className="grid md:grid-cols-2 gap-6">
            {projects.length > 0 ? (
              projects.map((project: any) => (
                <ProjectCard
                  key={project.slug}
                  project={project}
                  progress={progressMap[project.slug]}
                />
              ))
            ) : (
              <div className="col-span-full bg-white/80 border border-gray-200 rounded-xl p-12 text-center shadow-sm backdrop-blur-sm">
                <div className="text-5xl mb-3">📚</div>
                <p className="text-gray-800 font-medium">No projects found.</p>
                <p className="text-gray-500 text-sm mt-1">
                  Check your <code className="bg-gray-100 px-1 rounded">/public/projects</code> folder.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Sidebar */}
        <aside className="lg:col-span-1">
          <div className="sticky top-28">
            <ProfileSidebar
              user={user}
              userProfile={userProfile}
              createdAt={userProfile?.created_at}
              projectsEnrolled={projectsEnrolled}
              completedSteps={completedSteps}
              projectsInProgress={projectsInProgress}
            />
          </div>
        </aside>
      </main>

      {/* Footer */}
      <footer
        className="border-t mt-16 py-8 backdrop-blur-md"
        style={{
          backgroundColor: `${lavender}B3`,
          borderColor: mint,
        }}
      >
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-700 text-sm">
            © 2025 CrekAI — Empowering learners through hands-on AI experiences.
          </p>
          <div className="flex gap-5 text-sm">
            <a href="#" className="text-gray-700 hover:text-gray-900 transition">
              About
            </a>
            <a href="#" className="text-gray-700 hover:text-gray-900 transition">
              Help
            </a>
            <a href="#" className="text-gray-700 hover:text-gray-900 transition">
              Contact
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}
