"use server"

import { createServerClient } from "@supabase/ssr"
import { cookies } from "next/headers"

export async function signUpAction(email: string, password: string, username: string, fullName: string) {
  const cookieStore = await cookies()
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => cookieStore.set(name, value, options))
          } catch {
            // Handle error
          }
        },
      },
    },
  )

  const { data: authData, error: authError } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: `${process.env.NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL || process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "http://localhost:3000"}/projects`,
    },
  })

  if (authError || !authData.user) {
    return { error: authError?.message || "Sign up failed" }
  }

  const serviceSupabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => cookieStore.set(name, value, options))
          } catch {
            // Handle error
          }
        },
      },
    },
  )

  const { error: profileError } = await serviceSupabase.from("user_profiles").insert({
    id: authData.user.id,
    email,
    username,
    full_name: fullName,
  })

  if (profileError) {
    return { error: profileError.message }
  }

  return { data: authData.user }
}
