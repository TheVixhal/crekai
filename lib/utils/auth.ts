import { createClient } from "@/lib/supabase/client"

export async function signIn(email: string, password: string) {
  const supabase = createClient()

  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  })

  if (error) {
    return { error: error.message }
  }

  return { data: data.user }
}

export async function signOut() {
  const supabase = createClient()
  return supabase.auth.signOut()
}
