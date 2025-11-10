import { createServerClient } from "@supabase/ssr"
import { cookies } from "next/headers"
import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get("code")
  const origin = requestUrl.origin

  if (code) {
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
              cookiesToSet.forEach(({ name, value, options }) => 
                cookieStore.set(name, value, options)
              )
            } catch {
              // Handle error
            }
          },
        },
      }
    )

    const { data: { user }, error: exchangeError } = await supabase.auth.exchangeCodeForSession(code)

    if (!exchangeError && user) {
      // Check if user profile already exists
      const { data: existingProfile } = await supabase
        .from("user_profiles")
        .select("id")
        .eq("id", user.id)
        .single()

      // Create profile if it doesn't exist (for new Google sign-ups)
      if (!existingProfile) {
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
                  cookiesToSet.forEach(({ name, value, options }) => 
                    cookieStore.set(name, value, options)
                  )
                } catch {
                  // Handle error
                }
              },
            },
          }
        )

        // Extract username from email (before @) or use user metadata
        const username = user.user_metadata?.username || 
          user.email?.split("@")[0] || 
          `user_${user.id.slice(0, 8)}`

        const fullName = user.user_metadata?.full_name || 
          user.user_metadata?.name || 
          ""

        await serviceSupabase.from("user_profiles").insert({
          id: user.id,
          email: user.email!,
          username,
          full_name: fullName,
        })
      }

      // Redirect to projects page after successful authentication
      return NextResponse.redirect(`${origin}/projects`)
    }
  }

  // Return the user to an error page with instructions
  return NextResponse.redirect(`${origin}/auth/auth-error`)
}
