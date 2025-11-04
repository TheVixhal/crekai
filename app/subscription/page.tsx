import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import Image from "next/image"
import Link from "next/link"

export default async function SubscriptionPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth/login")
  }

  const { data: userProfile } = await supabase
    .from("user_profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle()

  const userLevel = userProfile?.level ?? 0

  return (
    <div className="min-h-screen bg-[#F7F5F2]">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <Link href="/projects" className="flex items-center gap-3">
            <Image 
              src="/sphereo.png" 
              alt="CrekAI Logo" 
              width={40} 
              height={40}
              className="rounded-lg"
            />
            <h1 className="text-2xl font-semibold text-gray-900">CrekAI</h1>
          </Link>
          
          <Link 
            href="/projects"
            className="px-5 py-2 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 transition-colors"
          >
            Back to Projects
          </Link>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-4xl mx-auto p-6 lg:p-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-semibold text-gray-900 mb-4">
            Upgrade to Pro
          </h2>
          <p className="text-lg text-gray-600">
            Unlock all projects and learning modules
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Free Plan */}
          <div className="bg-white rounded-xl border-2 border-gray-200 p-8">
            <div className="mb-6">
              <h3 className="text-2xl font-semibold text-gray-900 mb-2">Free</h3>
              <div className="flex items-baseline gap-2 mb-4">
                <span className="text-4xl font-bold text-gray-900">$0</span>
                <span className="text-gray-500">/month</span>
              </div>
            </div>

            <ul className="space-y-3 mb-8">
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-gray-400 mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span className="text-gray-600">Access to free projects</span>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-gray-400 mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span className="text-gray-600">Basic learning modules</span>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-gray-400 mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span className="text-gray-600">Community support</span>
              </li>
            </ul>

            {userLevel === 0 && (
              <div className="px-6 py-3 bg-gray-100 text-gray-700 rounded-lg font-medium text-center">
                Current Plan
              </div>
            )}
          </div>

          {/* Pro Plan */}
          <div className="bg-black text-white rounded-xl border-2 border-black p-8 relative">
            <div className="absolute -top-4 left-1/2 -translate-x-1/2">
              <span className="bg-white text-black px-4 py-1 rounded-full text-sm font-semibold">
                Popular
              </span>
            </div>

            <div className="mb-6">
              <h3 className="text-2xl font-semibold mb-2">Pro</h3>
              <div className="flex items-baseline gap-2 mb-4">
                <span className="text-4xl font-bold">$29</span>
                <span className="text-gray-300">/month</span>
              </div>
            </div>

            <ul className="space-y-3 mb-8">
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-white mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span className="text-gray-100">All free features</span>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-white mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span className="text-gray-100">Access to all premium projects</span>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-white mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span className="text-gray-100">Advanced learning modules</span>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-white mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span className="text-gray-100">Priority support</span>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-white mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span className="text-gray-100">Exclusive content & updates</span>
              </li>
            </ul>

            {userLevel === 0 ? (
              <button className="w-full px-6 py-3 bg-white text-black rounded-lg font-medium hover:bg-gray-100 transition-colors">
                Upgrade to Pro
              </button>
            ) : (
              <div className="w-full px-6 py-3 bg-white/10 text-white rounded-lg font-medium text-center">
                Current Plan
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
