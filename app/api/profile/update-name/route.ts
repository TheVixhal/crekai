import { createClient } from "@/lib/supabase/server"
import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const { fullName } = await request.json()

    if (!fullName || !fullName.trim()) {
      return NextResponse.json(
        { error: "Name is required" },
        { status: 400 }
      )
    }

    const supabase = await createClient()
    const {
      data: { user },
    } = await supabase.auth.getUser()

    if (!user) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      )
    }

    // Update user profile
    const { error } = await supabase
      .from("user_profiles")
      .update({ full_name: fullName.trim() })
      .eq("id", user.id)

    if (error) {
      console.error("Error updating name:", error)
      return NextResponse.json(
        { error: "Failed to update name" },
        { status: 500 }
      )
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error in update-name API:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
