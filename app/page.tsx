import Link from "next/link"
import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import TypewriterHeading from "@/components/TypewriterHeading"
import CrackEffect from "@/components/CrackEffect"
import { FloatingNav } from "@/components/ui/floating-navbar";

export default async function Home() {
    const navItems = [
    {
      name: "Home",
      link: "/",
      icon: <IconHome className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },
    {
      name: "About",
      link: "/about",
      icon: <IconUser className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },
    {
      name: "Contact",
      link: "/contact",
      icon: (
        <IconMessage className="h-4 w-4 text-neutral-500 dark:text-white" />
      ),
    },
  ];
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (user) {
    redirect("/projects")
  }
  return (
    <CrackEffect>
    <div className="relative  w-full">  
    <FloatingNav navItems={navItems} />  
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-amber-50 via-orange-50 to-amber-100">
      {/* Main Masthead */}
      <div className="border-b-4 border-black py-12 px-6 text-center bg-gradient-to-b from-white to-gray-50 shadow-lg relative overflow-hidden rounded-b-3xl">
        <div className="absolute inset-0 opacity-5" style={{
          backgroundImage: `repeating-linear-gradient(0deg, #000 0px, #000 1px, transparent 1px, transparent 4px),
                           repeating-linear-gradient(90deg, #000 0px, #000 1px, transparent 1px, transparent 4px)`
        }}></div>
        <div className="relative z-10">
          <h1 className="text-7xl md:text-8xl font-bold text-black font-serif tracking-wider mb-4 drop-shadow-sm">Crek AI</h1>
          <div className="h-2 bg-gradient-to-r from-transparent via-black to-transparent mx-auto w-64 mb-4 shadow-md"></div>
          <p className="text-lg md:text-xl text-gray-800 font-serif italic mb-2">
            Learn AI & Machine Learning Through Hands On Practice
          </p>
          <p className="text-xs text-gray-600 tracking-widest font-sans uppercase">Your Source for AI/ML Mastery</p>
        </div>
      </div>

      {/* Hero Section */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 p-6 max-w-7xl mx-auto w-full">
          {/* Main Featured Article */}
          <div>
            <div className="bg-white border-2 border-black h-full flex flex-col shadow-2xl hover:shadow-3xl transition-shadow duration-300 relative overflow-hidden rounded-3xl">
              <div className="absolute top-0 right-0 w-32 h-32 bg-amber-100 opacity-20 rounded-full blur-3xl"></div>
              <div className="absolute bottom-0 left-0 w-40 h-40 bg-orange-100 opacity-20 rounded-full blur-3xl"></div>
              
              <div className="border-b-2 border-black p-6 bg-gradient-to-r from-gray-50 to-white relative">
                <p className="text-xs uppercase tracking-widest font-sans font-bold mb-2 text-orange-600">Featured</p>
                <TypewriterHeading />
                <p className="text-sm text-gray-700 font-sans">
                  Master machine learning fundamentals through hands-on projects
                </p>
              </div>
              
              <div className="p-6 flex-1 flex flex-col justify-between relative z-10">
                <div>
                  <p className="text-lg font-serif text-black mb-4 leading-relaxed">
                    CrekAI brings you industry-standard AI and ML courses built for real-world learning. Learn from
                    expert projects, solve real problems, and unlock your potential in artificial intelligence.
                  </p>
                  <div className="space-y-3 mb-6">
                    <div className="flex items-start group">
                      <span className="font-bold text-orange-600 mr-3 group-hover:translate-x-1 transition-transform">→</span>
                      <p className="text-black font-sans">Learn by building real AI projects, not watching videos</p>
                    </div>
                    <div className="flex items-start group">
                      <span className="font-bold text-orange-600 mr-3 group-hover:translate-x-1 transition-transform">→</span>
                      <p className="text-black font-sans">Progress through structured, step-by-step learning paths</p>
                    </div>
                    <div className="flex items-start group">
                      <span className="font-bold text-orange-600 mr-3 group-hover:translate-x-1 transition-transform">→</span>
                      <p className="text-black font-sans">Track your progress and unlock achievements</p>
                    </div>
                  </div>
                </div>
                
                <div>
                  <Link href="/auth/sign-up">
                  <button className="w-full px-8 py-4 bg-gradient-to-r from-orange-500 to-amber-500 text-white font-serif text-lg font-bold border-2 border-black hover:from-orange-600 hover:to-amber-600 transition-all shadow-md hover:shadow-xl hover:translate-y-[-2px]">
                    Try Now
                  </button>
                  </Link>

                  <Link href="/auth/login">
                  <button className="w-full mt-4 px-8 py-4 bg-white text-black font-serif text-lg font-bold border-2 border-black hover:bg-gray-100 transition-all shadow-sm hover:shadow-lg hover:translate-y-[-2px]">
                    Log In
                  </button>
                  </Link>  

                    
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section - Learning Paths Preview */}
        <div className="border-t-4 border-black bg-gradient-to-b from-white to-gray-50 mt-6 shadow-inner">
          <div className="max-w-7xl mx-auto px-6 py-12">
            <h3 className="text-3xl font-bold font-serif text-black mb-8">Popular Projects</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="border-2 border-black p-6 bg-white hover:shadow-2xl transition-all duration-300 hover:translate-y-[-4px] group relative overflow-hidden rounded">
                <div className="absolute top-0 right-0 w-24 h-24 bg-blue-100 opacity-0 group-hover:opacity-30 rounded-full blur-2xl transition-opacity"></div>
                <p className="text-xs uppercase tracking-widest font-bold mb-3 text-blue-600 font-sans relative z-10">Beginner</p>
                <h4 className="text-xl font-bold font-serif mb-3 text-black relative z-10">Build Artificial Neural Network</h4>
                <p className="text-sm text-gray-700 mb-4 relative z-10">
                  An artificial neural network (ANN) is a computational model inspired by the human brain's structure and function, consisting of interconnected nodes called neurons.
                </p>
                <p className="text-xs text-gray-600 font-sans relative z-10">15 Steps • Beginner Friendly</p>
              </div>
              
              <div className="border-2 border-black p-6 bg-white hover:shadow-2xl transition-all duration-300 hover:translate-y-[-4px] group relative overflow-hidden rounded">
                <div className="absolute top-0 right-0 w-24 h-24 bg-purple-100 opacity-0 group-hover:opacity-30 rounded-full blur-2xl transition-opacity"></div>
                <p className="text-xs uppercase tracking-widest font-bold mb-3 text-purple-600 font-sans relative z-10">Intermediate</p>
                <h4 className="text-xl font-bold font-serif mb-3 text-black relative z-10">Build Tokeniser</h4>
                <p className="text-sm text-gray-700 mb-4 relative z-10">
                  A tokenizer in Natural Language Processing (NLP) is a tool that breaks down raw text into smaller, manageable units called tokens.
                </p>
                <p className="text-xs text-gray-600 font-sans relative z-10">20 Steps • Intermediate Project</p>
              </div>
              
              <div className="border-2 border-black p-6 bg-white hover:shadow-2xl transition-all duration-300 hover:translate-y-[-4px] group relative overflow-hidden rounded">
                <div className="absolute top-0 right-0 w-24 h-24 bg-green-100 opacity-0 group-hover:opacity-30 rounded-full blur-2xl transition-opacity"></div>
                <p className="text-xs uppercase tracking-widest font-bold mb-3 text-green-600 font-sans relative z-10">Advanced</p>
                <h4 className="text-xl font-bold font-serif mb-3 text-black relative z-10">Build Your Own GPT</h4>
                <p className="text-sm text-gray-700 mb-4 relative z-10">
                 GPT stands for Generative Pre-trained Transformer, which is a type of language model that uses a neural network architecture called a transformer to generate human-like text.
                </p>
                <p className="text-xs text-gray-600 font-sans relative z-10">25+ Steps • Advanced Project</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t-4 border-black bg-gradient-to-t from-gray-100 to-white text-black py-8 px-6 text-center">
        <p className="font-serif text-sm mb-2">CrekAI - Master AI & Machine Learning</p>
        <p className="text-xs text-gray-500">© 2025 All rights reserved</p>
      </div>
    </div>
    </div>   
  </CrackEffect>  
  )
}
