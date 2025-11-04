import { createClient } from "@/lib/supabase/server"
import { NextResponse } from "next/server"
import crypto from "crypto"

export async function POST() {
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

    // Generate a secure random token
    const token = crypto.randomBytes(32).toString("hex")

    // Update user profile with the new token
    const { error } = await supabase
      .from("user_profiles")
      .update({ colab_token: token })
      .eq("id", user.id)

    if (error) {
      console.error("Token generation error:", error)
      return NextResponse.json(
        { error: "Failed to generate token" },
        { status: 500 }
      )
    }

    return NextResponse.json({
      success: true,
      token,
    })
  } catch (error) {
    console.error("Generate token error:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
