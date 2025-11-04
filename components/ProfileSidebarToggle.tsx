"use client"

import { useEffect } from "react"

export default function ProfileSidebarToggle() {
  useEffect(() => {
    const profileToggle = document.getElementById("profile-toggle")
    const profileClose = document.getElementById("profile-close")
    const profileSidebar = document.getElementById("profile-sidebar")
    const profileOverlay = document.getElementById("profile-overlay")

    if (!profileToggle || !profileSidebar || !profileOverlay) return

    function openSidebar() {
      profileSidebar.classList.remove("translate-x-full")
      profileOverlay.classList.remove("opacity-0", "pointer-events-none")
    }

    function closeSidebar() {
      profileSidebar.classList.add("translate-x-full")
      profileOverlay.classList.add("opacity-0", "pointer-events-none")
    }

    profileToggle.addEventListener("click", openSidebar)
    profileClose?.addEventListener("click", closeSidebar)
    profileOverlay.addEventListener("click", closeSidebar)

    // Cleanup when unmounting
    return () => {
      profileToggle.removeEventListener("click", openSidebar)
      profileClose?.removeEventListener("click", closeSidebar)
      profileOverlay.removeEventListener("click", closeSidebar)
    }
  }, [])

  return null
}
