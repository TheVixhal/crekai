import type React from "react"
import { createClient } from "@/lib/supabase/server"
import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { signIn } from "@/lib/utils/auth"

export default async function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    const result = await signIn(email, password)

    if (result.error) {
      setError(result.error)
      setLoading(false)
    } else {
      router.push("/projects")
    }
  }

  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (user) {
    redirect("/projects")
  }

  return (
    <div className="min-h-screen bg-amber-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Newspaper header */}
        <div className="border-b-4 border-black mb-8 pb-4 text-center">
          <h1 className="text-4xl font-bold text-black font-serif">Crekai</h1>
          <p className="text-xs text-gray-700 uppercase tracking-wider mt-2">Learning Platform</p>
        </div>

        <div className="bg-white border-2 border-black p-8">
          <h2 className="text-2xl font-bold text-black mb-6 text-center font-serif">Sign In</h2>

          {error && (
            <div className="bg-red-100 border-2 border-red-800 text-red-900 px-4 py-3 mb-4 font-sans">{error}</div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-bold text-black mb-2 font-sans">EMAIL</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                required
                className="w-full border-2 border-black p-2 bg-white text-black placeholder:text-gray-500 font-sans"
              />
            </div>

            <div>
              <label className="block text-sm font-bold text-black mb-2 font-sans">PASSWORD</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                className="w-full border-2 border-black p-2 bg-white text-black placeholder:text-gray-500 font-sans"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-black text-white font-serif font-bold py-2 border-2 border-black hover:bg-gray-900 disabled:opacity-50 transition"
            >
              {loading ? "Signing in..." : "SIGN IN"}
            </button>
          </form>

          <div className="border-t-2 border-black mt-6 pt-6 text-center text-sm font-sans">
            <p className="text-black">
              Don't have an account?{" "}
              <Link href="/auth/sign-up" className="font-bold underline hover:no-underline">
                Create one
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
