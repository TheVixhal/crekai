import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter",
  display: 'swap',
})

export const metadata: Metadata = {
  title: "CrekAI - Learn AI/ML by Building",
  description: "Learn AI and Machine Learning through hands-on practice",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/sphereo.png" sizes="any" />
      </head>
      <body className={`${inter.variable} antialiased font-sans`}>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
