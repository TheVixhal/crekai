import Link from "next/link"
import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"

export default async function Home() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (user) {
    redirect("/projects")
  }

  return (
    <div className="min-h-screen flex flex-col bg-amber-50">
      {/* Main Masthead */}
      <div className="border-b-4 border-black py-12 px-6 text-center bg-white">
        <h1 className="text-7xl md:text-8xl font-bold text-black font-serif tracking-wider mb-4">CrekAI</h1>
        <div className="h-2 bg-black mx-auto w-64 mb-4"></div>
        <p className="text-lg md:text-xl text-gray-800 font-serif italic mb-2">
          Learn AI & Machine Learning Through Hands On Practice
        </p>
        <p className="text-xs text-gray-600 tracking-widest font-sans uppercase">Your Source for AI/ML Mastery</p>
      </div>

      {/* Hero Section */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 p-6 max-w-7xl mx-auto w-full">
          {/* Main Featured Article */}
          <div className>
            <div className="bg-white border-4 border-black h-full flex flex-col">
              <div className="border-b-2 border-black p-6">
                <p className="text-xs uppercase tracking-widest font-sans font-bold mb-2 text-gray-700">Featured</p>
                <h2 className="text-4xl md:text-5xl font-bold font-serif mb-3 text-black">
                  Start Your AI Journey Today
                </h2>
                <p className="text-sm text-gray-700 font-sans">
                  Master machine learning fundamentals through hands-on projects
                </p>
              </div>
              <div className="p-6 flex-1 flex flex-col justify-between">
                <div>
                  <p className="text-lg font-serif text-black mb-4 leading-relaxed">
                    CrekAI brings you industry-standard AI and ML courses built for real-world learning. Learn from
                    expert projects, solve real problems, and unlock your potential in artificial intelligence.
                  </p>
                  <div className="space-y-3 mb-6">
                    <div className="flex items-start">
                      <span className="font-bold text-black mr-3">→</span>
                      <p className="text-black font-sans">Learn by building real AI projects, not watching videos</p>
                    </div>
                    <div className="flex items-start">
                      <span className="font-bold text-black mr-3">→</span>
                      <p className="text-black font-sans">Progress through structured, step-by-step learning paths</p>
                    </div>
                    <div className="flex items-start">
                      <span className="font-bold text-black mr-3">→</span>
                      <p className="text-black font-sans">Track your progress and unlock achievements</p>
                    </div>
                  </div>
                </div>
                {/* FIX: You cannot nest <Link> components. 
                  They must be siblings.
                  I also changed border-3 to border-2 (a valid Tailwind class)
                  and added a margin-top (mt-4) to the second button.
                */}
                <div>
                  <Link href="/auth/sign-up">
                    <button className="w-full px-8 py-4 bg-white text-black font-serif text-lg font-bold border-2 border-black hover:bg-gray-50 transition">
                      Try Now
                    </button>
                  </Link>
                  <Link href="/auth/login">
                    <button className="w-full mt-4 px-8 py-4 bg-white text-black font-serif text-lg font-bold border-2 border-black hover:bg-gray-50 transition">
                      Log In
                    </button>
                  </Link>
                </div>
              </div>
            </div>
          </div>

          {/* This empty column (md:col-span-1) is fine, no errors here */}
        </div>

        {/* Bottom Section - Learning Paths Preview */}
        <div className="border-t-4 border-black bg-white mt-6">
          <div className="max-w-7xl mx-auto px-6 py-12">
            <h3 className="text-3xl font-bold font-serif text-black mb-8">Popular Learning Paths</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="border-2 border-black p-6 bg-white hover:shadow-lg transition">
                <p className="text-xs uppercase tracking-widest font-bold mb-3 text-gray-600 font-sans">Beginner</p>
                <h4 className="text-xl font-bold font-serif mb-3 text-black">AI Fundamentals</h4>
                <p className="text-sm text-gray-700 mb-4">
                  Start with the basics of artificial intelligence and machine learning concepts.
                </p>
                <p className="text-xs text-gray-600 font-sans">5 Projects • Beginner Friendly</p>
              </div>
              <div className="border-2 border-black p-6 bg-white hover:shadow-lg transition">
                <p className="text-xs uppercase tracking-widest font-bold mb-3 text-gray-600 font-sans">Intermediate</p>
                <h4 className="text-xl font-bold font-serif mb-3 text-black">Deep Learning</h4>
                <p className="text-sm text-gray-700 mb-4">
                  Dive deep into neural networks and build advanced AI models.
                </p>
                <p className="text-xs text-gray-600 font-sans">8 Projects • Build Real Models</p>
              </div>
              <div className="border-2 border-black p-6 bg-white hover:shadow-lg transition">
                <p className="text-xs uppercase tracking-widest font-bold mb-3 text-gray-600 font-sans">Advanced</p>
                <h4 className="text-xl font-bold font-serif mb-3 text-black">Production AI</h4>
                <p className="text-sm text-gray-700 mb-4">
                  Deploy AI models to production and scale your applications.
                </p>
                <p className="text-xs text-gray-600 font-sans">6 Projects • Deploy to Production</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t-4 border-black bg-white text-black py-8 px-6 text-center">
        <p className="font-serif text-sm mb-2">CrekAI - Master AI & Machine Learning</p>
        <p className="text-xs text-gray-500">© 2025 All rights reserved</p>
      </div>
    </div>
  )
}
