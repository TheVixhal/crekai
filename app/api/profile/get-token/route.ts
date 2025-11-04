import { createClient } from "@/lib/supabase/server"
import { NextResponse } from "next/server"

export async function GET() {
  try {
    const supabase = await createClient()
    
    // Get current user
    const {
      data: { user },
    } = await supabase.auth.getUser()

    if (!user) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      )
    }

    // Get user profile with token
    const { data: profile, error } = await supabase
      .from("user_profiles")
      .select("colab_token")
      .eq("id", user.id)
      .single()

    if (error) {
      console.error("Get token error:", error)
      return NextResponse.json(
        { error: "Failed to get token" },
        { status: 500 }
      )
    }

    return NextResponse.json({
      success: true,
      token: profile?.colab_token || null,
    })
  } catch (error) {
    console.error("Get token error:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
